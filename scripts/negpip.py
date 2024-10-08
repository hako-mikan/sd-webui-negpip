import gradio as gr
import torch
import re
import json
from torch import nn, einsum
from einops import rearrange, repeat

try:
    import ldm.modules.attention as atm
    forge = False
except:
    #forge
    forge = True
    from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects

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
        self.isxl = hasattr(shared.sd_model,"conditioner")
        
        self.rev = p.sampler_name in ["DDIM", "PLMS", "UniPC"]
        if forge: self.rev = not self.rev

        if forge:
            tokenizer = p.sd_model.text_processing_engine.tokenize_line
        else:
            tokenizer = shared.sd_model.conditioner.embedders[0].tokenize_line if self.isxl else shared.sd_model.cond_stage_model.tokenize_line

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

        def conddealer(targets):
            conds =[]
            start = None
            end = None
            for target in targets:
                input = SdConditioning([f"({target[0]}:{-target[1]})"], width=p.width, height=p.height)
                with devices.autocast():
                    cond = prompt_parser.get_learned_conditioning(shared.sd_model,input,p.steps)
                if start is None: start = cond[0][0].cond[0:1,:] if not self.isxl else cond[0][0].cond["crossattn"][0:1,:]
                if end is None: end = cond[0][0].cond[-1:,:] if not self.isxl else cond[0][0].cond["crossattn"][-1:,:]
                token, tokenlen = tokenizer(target[0])
                conds.append(cond[0][0].cond[1:tokenlen +2,:] if not self.isxl else cond[0][0].cond["crossattn"][1:tokenlen +2,:] ) 
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
                self.handle = hook_forwards(self, p.sd_model.forge_objects.unet.model)
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

from pprint import pprint

def unload(self,p):
    if hasattr(self,"handle"):
        if forge:
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
