import gradio as gr
import torch
import re
import json
from torch import nn, einsum
from einops import rearrange, repeat
from inspect import isfunction

try:
    import ldm.modules.attention as atm
    forge = False
except:
    #forge
    forge = True
    from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
    from backend.nn.flux import attention, fp16_fix

import modules.ui
import modules
from modules import prompt_parser, devices
from modules import shared
from modules.script_callbacks import CFGDenoiserParams, on_cfg_denoiser, on_ui_settings
from modules.ui_components import InputAccordion

debug = False
debug_p = False

OPT_ACT = "negpip_active"
OPT_HIDE = "negpip_hide"

NEGPIP_T = "customscript/negpip.py/txt2img/Active/value"
NEGPIP_I = "customscript/negpip.py/img2img/Active/value"
CONFIG = shared.cmd_opts.ui_config_file

with open(CONFIG, 'r', encoding="utf-8") as json_file:
    ui_config = json.load(json_file)

startup_t = ui_config[NEGPIP_T] if NEGPIP_T in ui_config else None
startup_i = ui_config[NEGPIP_I] if NEGPIP_I in ui_config else None
active_t = "Active" if startup_t else "Not Active"
active_i = "Active" if startup_i else "Not Active"

opt_active = getattr(shared.opts,OPT_ACT, True)
opt_hideui = getattr(shared.opts,OPT_HIDE, False)

minusgetter = r'\(([^(:)]*):\s*-[\d]+(\.[\d]+)?(?:\s*)\)'

COND_KEY_C = "crossattn"
COND_KEY_V = "vector"

class Script(modules.scripts.Script):   
    def __init__(self):
        self.active = False
        self.conds = None
        self.unconds = None
        self.conlen = []   
        self.unlen = []
        self.contokens = []   
        self.untokens = []
        self.hr = False
        self.x = None

        self.ipa = None

        self.enable_rp_latent = False
        
    def title(self):
        return "NegPiP"

    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible

    infotext_fields = None
    paste_field_names = []

    def ui(self, is_img2img):    
        with InputAccordion(False, label=self.title()) as active:
            toggle = gr.Button(elem_id="switch_default", value=f"Toggle startup with Active(Now:{startup_i if is_img2img else startup_t})", variant="primary")

        def f_toggle(is_img2img):
            key = NEGPIP_I if is_img2img else NEGPIP_T

            with open(CONFIG, 'r', encoding="utf-8") as json_file:
                data = json.load(json_file)
            data[key] = not data[key]

            with open(CONFIG, 'w', encoding="utf-8") as json_file:
                json.dump(data, json_file, indent=4) 

            return gr.update(value = f"Toggle startup Active(Now:{data[key]})")

        toggle.click(fn=f_toggle,inputs=[gr.Checkbox(value = is_img2img, visible = False)],outputs=[toggle])
        active.change(fn=lambda x:gr.update(label = f"NegPiP : {'Active' if x else 'Not Active'}"),inputs=active, outputs=[active])

        self.infotext_fields = [
                (active, "NegPiP Active"),
        ]

        for _,name in self.infotext_fields:
            self.paste_field_names.append(name)

        return [active]

    def process_batch(self, p, active,**kwargs):
        self.__init__()
        flag = False

        if getattr(shared.opts,OPT_HIDE, False) and not getattr(shared.opts,OPT_ACT, False): return
        elif not active: return

        self.rpscript = None
        #get infomation of regponal prompter
        from modules.scripts import scripts_txt2img
        for script in scripts_txt2img.alwayson_scripts:
            if "rp.py" in script.filename:
                self.rpscript = script

        self.hrp, self.hrn = hr_dealer(p)

        self.active = active
        self.batch = p.batch_size
        # old call, hasattr(shared.sd_model,"conditioner"), no longer works on forge backend, but forge backend provides its own way to do this.
        self.isxl = p.sd_model.is_sdxl
        
        # if you want to change other things to be more mnemonic to the current backend, here's the pprint calls i used to figure all this out in my initial port.
        # lllyasviel should really document this stuff. it's a nice backend! but he hasn't told any of us how to use it.
        #pprint(dir(p))
        #pprint(dir(p.sd_model))
        #pprint(dir(p.sd_model.forge_objects.unet))
        #pprint(dir(p.sd_model.forge_objects.clip))
        #pprint(dir(p.sd_model.forge_objects.clip.tokenizer))
        #pprint(p.sd_model.is_sdxl)
        
        self.rev = p.sampler_name in ["DDIM", "PLMS", "UniPC"]
        if forge: self.rev = not self.rev

        if forge:
            if hasattr(p.sd_model, "text_processing_engine_l"):
                tokenizer = p.sd_model.text_processing_engine_l.tokenize_line
            else:
                tokenizer = p.sd_model.text_processing_engine.tokenize_line
            self.flux = flux = "flux" in str(type(p.sd_model.forge_objects.unet.model.diffusion_model))
        else:
            tokenizer = shared.sd_model.conditioner.embedders[0].tokenize_line if self.isxl else shared.sd_model.cond_stage_model.tokenize_line
            self.flux = flux = False

        def getshedulednegs(scheduled,prompts):
            output = []
            nonlocal flag
            for i, batch_shedule in enumerate(scheduled):
                stepout = []
                seps = None
                if self.rpscript:
                    if hasattr(self.rpscript,"seps"):
                        seps = self.rpscript.seps
                    self.enable_rp_latent = seps == "AND"

                for step,prompt in batch_shedule:
                    sep_prompts = prompt.split(seps) if seps else [prompt]
                    padd = 0
                    padtextweight = []
                    for sep_prompt in sep_prompts:
                        minusmatches = re.finditer(minusgetter, sep_prompt)
                        minus_targets = []
                        textweights = []
                        for minusmatch in minusmatches:
                            minus_targets.append(minusmatch.group().replace("(","").replace(")",""))

                            prompts[i] = prompts[i].replace(minusmatch.group(),"")
                        minus_targets = [x.split(":") for x in minus_targets]
                        #print(minus_targets)
                        for text,weight in minus_targets:
                            weight = float(weight)
                            if text == "BREAK": continue
                            if weight < 0:
                                textweights.append([text,weight])
                                flag = True
                        padtextweight.append([padd,textweights])
                        tokens, tokensnum = tokenizer(sep_prompt)
                        padd = tokensnum // 75 + 1 + padd
                    stepout.append([step,padtextweight])
                output.append(stepout)
            return output
        
        scheduled_p = prompt_parser.get_learned_conditioning_prompt_schedules(p.prompts,p.steps)
        scheduled_np = prompt_parser.get_learned_conditioning_prompt_schedules(p.negative_prompts,p.steps)

        if self.hrp: scheduled_hr_p = prompt_parser.get_learned_conditioning_prompt_schedules(p.hr_prompts,p.hr_second_pass_steps if p.hr_second_pass_steps > 0 else p.steps)
        if self.hrn: scheduled_hr_np = prompt_parser.get_learned_conditioning_prompt_schedules(p.hr_negative_prompts,p.hr_second_pass_steps if p.hr_second_pass_steps > 0 else p.steps)

        nip = getshedulednegs(scheduled_p,p.prompts)
        pin = getshedulednegs(scheduled_np,p.negative_prompts)

        if self.hrp: hr_nip = getshedulednegs(scheduled_hr_p,p.hr_prompts)
        if self.hrn: hr_pin = getshedulednegs(scheduled_hr_np,p.hr_negative_prompts)

        cond_key = COND_KEY_C

        def conddealer(targets):
            conds =[]
            start = None
            end = None
            if flux:
                for target in targets:
                    input = SdConditioning([f"({target[0]}:{-target[1]})"], width=p.width, height=p.height)
                    with devices.autocast():
                        cond = prompt_parser.get_learned_conditioning(shared.sd_model,input,p.steps)
                    cond_data = cond[0][0].cond
                    token, tokenlen = tokenizer(target[0])
                    conds.append(cond_data[cond_key][0:tokenlen + 1, :]) 
                conds = torch.cat(conds,0).unsqueeze(0)
                conds = conds.repeat(self.batch,1,1)
                return conds, conds.shape[1]
                
            for target in targets:
                input = SdConditioning([f"({target[0]}:{-target[1]})"], width=p.width, height=p.height)
                with devices.autocast():
                    cond = prompt_parser.get_learned_conditioning(shared.sd_model,input,p.steps)
                cond_data = cond[0][0].cond
                if start is None: start = cond_data[0:1, :] if not self.isxl else cond_data[cond_key][0:1, :]
                if end is None: end = cond_data[-1:, :] if not self.isxl else cond_data[cond_key][-1:, :]
                token, tokenlen = tokenizer(target[0])
                conds.append(cond_data[1:tokenlen +2,:] if not self.isxl else cond_data[cond_key][1:tokenlen +2, :]) 
            conds = torch.cat(conds, 0)

            conds = torch.split(conds, 75, dim=0)
            condsout = []
            condcount = []
            for cond in conds:
                condcount.append(cond.shape[0])
                repeat = 0 if cond.shape[0] == 75 else 75 - cond.shape[0]
                cond = torch.cat((start,cond,end.repeat(repeat + 1,1)),0)
                condsout.append(cond)
            condout = torch.cat(condsout,0).unsqueeze(0)
            return condout.repeat(self.batch,1,1), condcount

        def calcconds(targetlist):
            outconds = []
            for batch in targetlist:
                stepconds = []
                for step, regions in batch:
                    regionconds = []
                    for region, targets in regions:
                        if targets:
                            conds, contokens = conddealer(targets)
                            regionconds.append([region, conds, contokens])
                        else:
                            regionconds.append([region, None, None])
                    stepconds.append([step,regionconds])
                outconds.append(stepconds)
            return outconds
            
        self.conds_all = calcconds(nip)
        self.unconds_all = calcconds(pin)

        if self.hrp: self.hr_conds_all = calcconds(hr_nip)
        if self.hrn: self.hr_unconds_all = calcconds(hr_pin)

        #print(self.conds_all)
        #print(self.unconds_all)

        resetpcache(p)
        
        def calcsets(A, B):
            return A // B if A % B == 0 else A // B + 1

        self.conlen = calcsets(tokenizer(p.prompts[0])[1],75)
        self.unlen = calcsets(tokenizer(p.negative_prompts[0])[1],75)

        if not flag:
            self.active = False
            unload(self,p)
            return   

        if not hasattr(self,"negpip_dr_callbacks"):
            self.negpip_dr_callbacks = on_cfg_denoiser(self.denoiser_callback)

        #disable hookforward if hookfoward in regional prompter is eanble. 
        #negpip operation is treated in regional prompter

        already_hooked = False
        if self.rpscript is not None and hasattr(self.rpscript,"hooked"):already_hooked = self.rpscript.hooked

        if not already_hooked:
            if forge:
                self.handle = hook_forwards_f(self, p.sd_model.forge_objects.unet.model)  if flux else hook_forwards(self, p.sd_model.forge_objects.unet.model) 
            else:
                self.handle = hook_forwards(self, p.sd_model.model.diffusion_model)

        print(f"NegPiP enable, Positive:{self.conds_all[0][0][1][0][2]},Negative:{self.unconds_all[0][0][1][0][2]}")

        p.extra_generation_params.update({
            "NegPiP Active":active,
        })

    def postprocess(self, p, processed, *args):
        unload(self,p)
        self.conds_all = None
        self.unconds_all = None
    
    def denoiser_callback(self, params: CFGDenoiserParams):
        if debug: print(params.text_cond.shape)
        if self.active:
            if self.x is None: self.x = params.x.shape
            if self.x != params.x.shape: self.hr = True

            self.latenti = 0 

            condslist = []
            tokenslist = []
            conds = self.hr_conds_all if self.hrp and self.hr else  self.conds_all
            if conds is not None:
                for step, regions in conds[0]:
                    if step >= params.sampling_step + 2:
                        for region, conds, tokens in regions:
                            condslist.append(conds)
                            tokenslist.append(tokens)
                            if debug: print(f"current:{params.sampling_step + 2},selected:{step}")
                        break
                self.conds = condslist
                self.contokens = tokenslist

            uncondslist = []
            untokenslist = []
            unconds = self.hr_unconds_all if self.hrn and self.hr else  self.unconds_all
            if unconds is not None:
                for step, regions  in unconds[0]:
                    if step >= params.sampling_step + 2:
                        for region, unconds, untokens in regions:
                            uncondslist.append(unconds)
                            untokenslist.append(untokens)
                            break

                self.unconds = uncondslist
                self.untokens = untokenslist
        
        if self.flux and self.conds:
            self.orig_tokens = params.text_cond[COND_KEY_C].shape[1]
            params.text_cond[COND_KEY_C] = torch.cat([params.text_cond[COND_KEY_C],self.conds[0]],1)

from pprint import pprint

def unload(self,p):
    if hasattr(self,"handle"):
        if forge:
            if self.flux:
                hook_forwards_f(self, p.sd_model.forge_objects.unet.model, remove=True)
            else:
                hook_forwards(self, p.sd_model.forge_objects.unet.model, remove=True)
        else:
            hook_forwards(self, p.sd_model.model.diffusion_model, remove=True)
        del self.handle

# helper functions from LDM
def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def hook_forward(self, module):
    def forward(x, context=None, mask=None, additional_tokens=None, n_times_crossframe_attn_in_self=0, value = None, transformer_options=None):
        if debug: print(" x.shape:",x.shape,"context.shape:",context.shape,"self.contokens",self.contokens,"self.untokens",self.untokens)
        
        def sub_forward(x, context, mask, additional_tokens, n_times_crossframe_attn_in_self,conds,contokens,unconds,untokens, latent = None):
            if debug: print(" x.shape[0]:",x.shape[0],"batch:",self.batch *2)
            
            if x.shape[0] == self.batch *2:
                if debug: print(" x.shape[0] == self.batch *2")
            
                if self.rev:
                    contn,contp = context.chunk(2)
                    ixn,ixp = x.chunk(2)
                else:
                    contp,contn =  context.chunk(2)
                    ixp,ixn = x.chunk(2)  #x[0:self.batch,:,:],x[self.batch:,:,:]
                
                if conds is not None:
                    if contp.shape[0] != conds.shape[0]:
                        conds = conds.expand(contp.shape[0],-1,-1)
                    contp = torch.cat((contp,conds),1)
                if unconds is not None:
                    if contn.shape[0] != unconds.shape[0]:
                        unconds = unconds.expand(contn.shape[0],-1,-1)
                    contn =  torch.cat((contn,unconds),1)
                xp = main_foward(self, module, ixp,contp,mask,additional_tokens,n_times_crossframe_attn_in_self,contokens)
                xn = main_foward(self, module, ixn,contn,mask,additional_tokens,n_times_crossframe_attn_in_self,untokens)
            
                out = torch.cat([xn,xp]) if self.rev else torch.cat([xp,xn])
                return out

            elif latent is not None:
                if debug:print(" latent is not None")
                if latent:
                    conds = conds if conds is not None else None
                else:
                    conds = unconds if unconds is not None else None
                if conds is not None:
                    if context.shape[0] != conds.shape[0]:
                        conds = conds.expand(context.shape[0],-1,-1)
                    context = torch.cat([context,conds],1)
                
                tokens = contokens if contokens is not None else untokens

                out = main_foward(self, module, x,context,mask,additional_tokens,n_times_crossframe_attn_in_self,tokens)
                return out

            else:
                if debug:
                    print(" Else")
                    print(context.shape[1] , self.conlen,self.unlen)

                tokens = []
                concon = counter(self.isxl)
                if debug: print(concon)
                if context.shape[1] == self.conlen * 77 and concon:
                    if conds is not None:
                        if context.shape[0] != conds.shape[0]:
                            conds = conds.expand(context.shape[0],-1,-1)
                        context = torch.cat([context,conds],1)
                        tokens = contokens
                elif context.shape[1] == self.unlen * 77 and concon:
                    if unconds is not None:
                        if context.shape[0] != unconds.shape[0]:
                            unconds = unconds.expand(context.shape[0],-1,-1)
                        context = torch.cat([context,unconds],1)
                        tokens = untokens
                out = main_foward(self, module, x,context,mask,additional_tokens,n_times_crossframe_attn_in_self,tokens)
                return out

        if self.enable_rp_latent:
            if len(self.conds) - 1 >= self.latenti:
                out = sub_forward(x, context, mask, additional_tokens, n_times_crossframe_attn_in_self,self.conds[self.latenti],self.contokens[self.latenti],None,None ,latent = True)
                self.latenti += 1
            else:
                out = sub_forward(x, context, mask, additional_tokens, n_times_crossframe_attn_in_self,None,None,self.unconds[0],self.untokens[0], latent = False)
                self.latenti = 0
            return out
        else:
            if self.conds is not None and self.unconds is not None and len(self.conds) > 0 and len(self.unconds) > 0:
                return sub_forward(x, context, mask, additional_tokens, n_times_crossframe_attn_in_self,self.conds[0],self.contokens[0],self.unconds[0],self.untokens[0])
            else:
                return sub_forward(x, context, mask, additional_tokens, n_times_crossframe_attn_in_self,None,None,None,None)
    
    return forward

count = 0
pn = True

def counter(isxl):
    global count, pn
    count += 1

    limit = 70 if isxl else 16
    outpn = pn

    if count == limit:
        pn = not pn
        count = 0
    return outpn

def main_foward(self, module, x, context, mask, additional_tokens, n_times_crossframe_attn_in_self, tokens):
    h = module.heads
    context = context.to(x.dtype)
    q = module.to_q(x)

    context = default(context, x)
    k = module.to_k(context)
    v = module.to_v(context)
    if debug: print(h,context.shape,q.shape,k.shape,v.shape)

    _, _, dim_head = q.shape
    dim_head //= h
    scale = dim_head ** -0.5

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
    sim = einsum('b i d, b j d -> b i j', q, k) * scale

    if self.active:
        if tokens:
            for token in tokens:
                start = (v.shape[1]//77 - len(tokens)) * 77
                #print("v.shape:",v.shape,"start:",start+1,"stop:",start+token)
                v[:,start+1:start+token,:] = -v[:,start+1:start+token,:] 

    if exists(mask):
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    attn = sim.softmax(dim=-1)
    #print(h,context.shape,q.shape,k.shape,v.shape,attn.shape)
    out = einsum('b i j, b j d -> b i d', attn, v)

    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

    return module.to_out(out)

import inspect

def hook_forwards(self, root_module: torch.nn.Module, remove=False):
    for name, module in root_module.named_modules():
        if "attn2" in name and module.__class__.__name__ == "CrossAttention":
            module.forward = hook_forward(self, module)
            if remove:
                del module.forward

def hook_forwards_f(self, root_module: torch.nn.Module, remove=False):
    for name, module in root_module.named_modules():
        if "double_blocks" in name and module.__class__.__name__ == "DoubleStreamBlock":
            module.forward = hook_forward_f_d(self, module)
            if remove:
                del module.forward

        if "single_blocks" in name and module.__class__.__name__ == "SingleStreamBlock":
                    module.forward = hook_forward_f_s(self, module)
                    if remove:
                        del module.forward

def resetpcache(p):
    p.cached_c = [None,None]
    p.cached_uc = [None,None]
    p.cached_hr_c = [None, None]
    p.cached_hr_uc = [None, None]


class SdConditioning(list):
    def __init__(self, prompts, is_negative_prompt=False, width=None, height=None, copy_from=None):
        super().__init__()
        self.extend(prompts)

        if copy_from is None:
            copy_from = prompts

        self.is_negative_prompt = is_negative_prompt or getattr(copy_from, 'is_negative_prompt', False)
        self.width = width or getattr(copy_from, 'width', None)
        self.height = height or getattr(copy_from, 'height', None)

def ext_on_ui_settings():
    # [setting_name], [default], [label], [component(blank is checkbox)], [component_args]debug_level_choices = []
    negpip_options = [
        (OPT_HIDE, False, "Hide in Txt2Img/Img2Img tab(Reload UI required)"),
        (OPT_ACT, True, "Active(Effective when Hide is Checked)",),
    ]
    section = ('negpip', "NegPiP")

    for cur_setting_name, *option_info in negpip_options:
        shared.opts.add_option(cur_setting_name, shared.OptionInfo(*option_info, section=section))

on_ui_settings(ext_on_ui_settings)

def hr_dealer(p):
    if not hasattr(p, "hr_prompts"):
        p.hr_prompts = None
    if not hasattr(p, "hr_negative_prompts"):
        p.hr_negative_prompts = None

    return bool(p.hr_prompts), bool(p.hr_negative_prompts )

def hook_forward_f_d(self, module):
    def double_s_forward(img, txt, vec, pe):
        img_mod1_shift, img_mod1_scale, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate = module.img_mod(vec)

        img_modulated = module.img_norm1(img)
        img_modulated = (1 + img_mod1_scale) * img_modulated + img_mod1_shift
        del img_mod1_shift, img_mod1_scale
        img_qkv = module.img_attn.qkv(img_modulated)
        del img_modulated

        # img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        B, L, _ = img_qkv.shape
        H = module.num_heads
        D = img_qkv.shape[-1] // (3 * H)
        img_q, img_k, img_v = img_qkv.view(B, L, 3, H, D).permute(2, 0, 3, 1, 4)
        del img_qkv

        img_q, img_k = module.img_attn.norm(img_q, img_k, img_v)

        txt_mod1_shift, txt_mod1_scale, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = module.txt_mod(vec)
        del vec

        txt_modulated = module.txt_norm1(txt)

        txt_modulated = (1 + txt_mod1_scale) * txt_modulated + txt_mod1_shift
        
        del txt_mod1_shift, txt_mod1_scale
        txt_qkv = module.txt_attn.qkv(txt_modulated)
        del txt_modulated

        B, L, _ = txt_qkv.shape
        txt_q, txt_k, txt_v = txt_qkv.view(B, L, 3, H, D).permute(2, 0, 3, 1, 4)
        del txt_qkv

        if self.contokens:
            txt_v[:,:,self.orig_tokens:self.orig_tokens + self.contokens[0],:] = -txt_v[:,:,self.orig_tokens:self.orig_tokens + self.contokens[0],:] 

        txt_q, txt_k = module.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        del txt_q, img_q
        k = torch.cat((txt_k, img_k), dim=2)
        del txt_k, img_k
        v = torch.cat((txt_v, img_v), dim=2)
        del txt_v, img_v

        attn = attention(q, k, v, pe=pe)
        del pe, q, k, v
        txt_attn, img_attn = attn[:, :txt.shape[1]], attn[:, txt.shape[1]:]

        del attn

        img = img + img_mod1_gate * module.img_attn.proj(img_attn)
        del img_attn, img_mod1_gate
        img = img + img_mod2_gate * module.img_mlp((1 + img_mod2_scale) * module.img_norm2(img) + img_mod2_shift)
        del img_mod2_gate, img_mod2_scale, img_mod2_shift

        txt = txt + txt_mod1_gate * module.txt_attn.proj(txt_attn)
        del txt_attn, txt_mod1_gate
        txt = txt + txt_mod2_gate * module.txt_mlp((1 + txt_mod2_scale) * module.txt_norm2(txt) + txt_mod2_shift)
        del txt_mod2_gate, txt_mod2_scale, txt_mod2_shift

        txt = fp16_fix(txt)

        return img, txt
    
    return double_s_forward


def hook_forward_f_s(self, module):
    def single_s_forward(x, vec, pe):
            mod_shift, mod_scale, mod_gate = module.modulation(vec)
            del vec
            x_mod = (1 + mod_scale) * module.pre_norm(x) + mod_shift
            del mod_shift, mod_scale
            qkv, mlp = torch.split(module.linear1(x_mod), [3 * module.hidden_size, module.mlp_hidden_dim], dim=-1)
            del x_mod

            # q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
            qkv = qkv.view(qkv.size(0), qkv.size(1), 3, module.num_heads, module.hidden_size // module.num_heads)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            del qkv
            if self.contokens:
                v[:,:,self.orig_tokens:self.orig_tokens + self.contokens[0],:] = -v[:,:,self.orig_tokens:self.orig_tokens + self.contokens[0],:]
            q, k = module.norm(q, k, v)

            attn = attention(q, k, v, pe=pe)
            del q, k, v, pe
            output = module.linear2(torch.cat((attn, module.mlp_act(mlp)), dim=2))
            del attn, mlp

            x = x + mod_gate * output
            del mod_gate, output

            x = fp16_fix(x)

            return x
    
    return single_s_forward
