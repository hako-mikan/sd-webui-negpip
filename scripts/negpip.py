import os
import gradio as gr
import torch
import re
import json
from torch import nn, einsum
from einops import rearrange, repeat
from inspect import isfunction
from modules.ui import versions_html
from packaging import version
from functools import wraps

classic = forge = reforge = False
try:
    import ldm.modules.attention as atm
    forge = False
except:
    #forge
    try:
        from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
        from backend.nn.flux import attention, fp16_fix
        forge = True
    except:
        classic = True

reforge = "reForge" in versions_html()

import torch.nn.functional as F
import modules.ui
import modules
from modules import prompt_parser, devices
from modules import shared
from modules.script_callbacks import CFGDenoiserParams, on_cfg_denoiser, on_ui_settings

debug = False
debug_p = False
debug_arch = os.environ.get("NEGPIP_DEBUG", "") == "1"

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

# group 1 is the target text, group 2 the negative weight. Escaped brackets
# (\( and \)) have to stay part of the target, see issue #68
minusgetter = r'\(((?:[^(:)\\]|\\.)*):\s*(-\d+(?:\.\d+)?)\s*\)'

COND_KEY_C = "crossattn"
COND_KEY_V = "vector"

# ---- NegPiP : Forge Neo (new architectures) ----------------------------------
# maps ForgeDiffusionEngine class name -> NegPiP internal model type
MODELTYPE_FROM_CLASS = {
    "ZImage": "ZImage",
    "Anima": "Anima",
    "Krea2": "Krea",
}

# trailing chat template tokens of the Qwen3 / Qwen3-VL text processing engines
TEMPLATE_TAIL = 5

# model types whose text encoder is an LLM. Their conditioning is built without
# emphasis and the weight is applied as a multiplier on the value vectors, see
# negpip_strength(). Applying it through the emphasis instead does not work:
# "Original" renormalises the mean afterwards and cancels most of the weight
LLM_MODELS = ("ZImage", "Anima", "Krea")

# Z-Image keeps the chat template in its conditioning, and the first token of the
# Qwen3 hidden states is an attention sink with a norm about 25x the others. A
# single added token is diluted by it, so the weight needs a gain to land on the
# same scale as the other architectures
MODEL_GAIN = {"ZImage": 10.0}

def get_text_engine(sd_model):
    """Return the text processing engine of a Forge/Forge-Neo model.

    Forge Neo names the engine after its text encoder (``text_processing_engine_anima``,
    ``..._qwen``, ``..._gemma`` ...) so a fixed attribute name no longer works.
    """
    for name in ("text_processing_engine_l", "text_processing_engine"):
        engine = getattr(sd_model, name, None)
        if engine is not None:
            return engine
    for name in dir(sd_model):
        if name.startswith("text_processing_engine"):
            engine = getattr(sd_model, name, None)
            if engine is not None:
                return engine
    return None

def tokenize_wrapper(engine):
    """``tokenize_line`` returns ``(chunks, token_count)`` on CLIP based engines but
    only ``chunks`` on every LLM based engine of Forge Neo. Normalise both."""
    tokenize_line = engine.tokenize_line

    def tokenize(line):
        result = tokenize_line(line)
        if isinstance(result, tuple):
            return result
        count = 0
        for chunk in result:
            tokens = getattr(chunk, "tokens", None)
            if tokens is None:
                tokens = getattr(chunk, "qwen_tokens", None) or getattr(chunk, "t5_tokens", [])
            count += len(tokens)
        return result, count

    return tokenize

def trim_padding(cond_data):
    """Anima pads its conditioning with zeros up to 512 tokens, drop the padding."""
    valid = torch.nonzero(cond_data.reshape(cond_data.shape[0], -1).abs().sum(dim=-1) > 0)
    if valid.numel() == 0:
        return cond_data
    return cond_data[:int(valid.max()) + 1]
# ------------------------------------------------------------------------------

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

        self.hooked_modules = []
        
    def title(self):
        return "NegPiP"

    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible

    infotext_fields = None
    paste_field_names = []

    def ui(self, is_img2img):    
        with InputAccordion(startup_i if is_img2img else startup_t, label=self.title(), visible = not opt_hideui) as active:
            toggle = gr.Button(elem_id="switch_default", value=f"Toggle startup with Active(Now:{startup_i if is_img2img else startup_t}), Needs Restart to Apply", variant="primary")

        def f_toggle(is_img2img):
            key = NEGPIP_I if is_img2img else NEGPIP_T

            with open(CONFIG, 'r', encoding="utf-8") as json_file:
                data = json.load(json_file)
            data[key] = not data.get(key, False)

            with open(CONFIG, 'w', encoding="utf-8") as json_file:
                json.dump(data, json_file, indent=4) 

            return gr.update(value = f"Toggle startup Active(Now:{data[key]})")

        toggle.click(fn=f_toggle,inputs=[gr.Checkbox(value = is_img2img, visible = False)],outputs=[toggle])

        self.infotext_fields = [
                (active, "NegPiP Active"),
        ]

        for _,name in self.infotext_fields:
            self.paste_field_names.append(name)

        return [active]

    def process_batch(self, p, active,**kwargs):
        self.__init__()
        flag = False

        if getattr(shared.opts,OPT_HIDE, False):
            # the accordion is hidden, so its checkbox cannot be reached; the setting
            # decides instead of the (invisible and always default) checkbox
            active = getattr(shared.opts,OPT_ACT, False)
        if not active: return

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
        if forge or reforge or classic: self.rev =  not self.rev 
        self.modeltype = modeltype = "SD"
         
        if forge:
            engine = get_text_engine(p.sd_model)
            modelclass = type(p.sd_model).__name__
            if modelclass in MODELTYPE_FROM_CLASS:
                self.modeltype = modeltype = MODELTYPE_FROM_CLASS[modelclass]
                # warm up / make sure the text encoder is resident before tokenizing
                input = SdConditioning([""], width=p.width, height=p.height)
                engine(input)
            tokenizer = tokenize_wrapper(engine)
            if "flux" in str(type(p.sd_model.forge_objects.unet.model.diffusion_model)):
                self.modeltype = modeltype = "flux"

        else:
            tokenizer = tokenize_wrapper(shared.sd_model.conditioner.embedders[0] if self.isxl else shared.sd_model.cond_stage_model)

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
                            # take the captured groups, stripping every bracket would also
                            # remove the escaped ones inside the target
                            minus_targets.append([minusmatch.group(1), minusmatch.group(2)])
                            prompts[i] = prompts[i].replace(minusmatch.group(),"")
                        for text,weight in minus_targets:
                            weight = float(weight)
                            if text == "BREAK": continue
                            if weight <= 0:
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
            if modeltype == "flux":
                strength = []
                for target in targets:
                    input = SdConditioning([f"({target[0]}:{-target[1]})"], width=p.width, height=p.height)
                    with devices.autocast():
                        cond = prompt_parser.get_learned_conditioning(shared.sd_model,input,p.steps)
                    cond_data = cond[0][0].cond
                    token, tokenlen = tokenizer(target[0])
                    conds.append(cond_data[cond_key][0:tokenlen + 1, :]) 
                    strength.extend([target[1]]*(cond_data.shape[0]))
                conds = torch.cat(conds,0).unsqueeze(0)
                conds = conds.repeat(self.batch,1,1)
                self.strength = strength
                return conds, conds.shape[1]
               
            if modeltype == "ZImage":
                strength = []
                for target in targets:
                    input = SdConditioning([f"({target[0]}:1.0)"], width=p.width, height=p.height)
                    with devices.autocast():
                        cond = prompt_parser.get_learned_conditioning(shared.sd_model,input,p.steps)
                    cond_data = cond[0][0].cond
                    conds.append(cond_data[3:-5, :])
                    strength.extend([target[1]]*(cond_data.shape[0]-8))
                conds = torch.cat(conds,0).unsqueeze(0)
                conds = conds.repeat(self.batch,1,1)
                self.strength = strength
                return conds, conds.shape[1]
                
            if modeltype in ("Anima", "Krea"):
                strength = []
                for target in targets:
                    input = SdConditioning([f"({target[0]}:1.0)"], width=p.width, height=p.height)
                    with devices.autocast():
                        cond = prompt_parser.get_learned_conditioning(shared.sd_model,input,p.steps)
                    cond_data = cond[0][0].cond
                    if modeltype == "Anima":
                        if cond_data.dim() > 2 and cond_data.shape[0] == 1: cond_data = cond_data[0]
                        cond_data = trim_padding(cond_data)
                        # the last token is the T5 end-of-sequence token, it carries a large
                        # part of the attention mass and must not be negated
                        if cond_data.shape[0] > 1: cond_data = cond_data[:-1]
                    else:
                        # Krea2 keeps the trailing chat template tokens, the leading
                        # part is already removed by the text processing engine
                        if cond_data.shape[0] > TEMPLATE_TAIL: cond_data = cond_data[:-TEMPLATE_TAIL]
                    if debug_arch: print("NegPiP/arch: cond", modeltype, tuple(cond_data.shape))
                    conds.append(cond_data)
                    strength.extend([target[1]]*cond_data.shape[0])
                conds = torch.cat(conds,0).unsqueeze(0)
                conds = conds.repeat(self.batch,*([1] * (conds.dim() - 1)))
                self.strength = strength
                return conds, conds.shape[1]

            for target in targets:
                strength = []
                input = SdConditioning([f"({target[0]}:{-target[1]})"], width=p.width, height=p.height)
                with devices.autocast():
                    cond = prompt_parser.get_learned_conditioning(shared.sd_model,input,p.steps)
                cond_data = cond[0][0].cond
                if start is None: start = cond_data[0:1, :] if not self.isxl else cond_data[cond_key][0:1, :]
                if end is None: end = cond_data[-1:, :] if not self.isxl else cond_data[cond_key][-1:, :]
                token, tokenlen = tokenizer(target[0])
                conds.append(cond_data[1:tokenlen +2,:] if not self.isxl else cond_data[cond_key][1:tokenlen +2, :]) 
            conds = torch.cat(conds, 0)

            # conds = torch.split(conds, 75, dim=0)
            # condsout = []
            # condcount = []
            # for cond in conds:
            #     condcount.append(cond.shape[0])
            #     repeat = 0 if cond.shape[0] == 75 else 75 - cond.shape[0]
            #     cond = torch.cat((start,cond,end.repeat(repeat + 1,1)),0)
            #     condsout.append(cond)
            conds = conds.unsqueeze(0)
            return conds.repeat(self.batch,1,1), conds.shape[1]

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
            self.handle = True
            hook_target(self, current_diffusion_model(p.sd_model))

        print(f"NegPiP enable, Positive:{self.conds_all[0][0][1][0][2]},Negative:{self.unconds_all[0][0][1][0][2]}")

        p.extra_generation_params.update({
            "NegPiP Active":active,
        })

    def postprocess(self, p, processed, *args):
        unload(self,p)
        self.conds_all = None
        self.unconds_all = None
       
    def denoiser_callback(self, params: CFGDenoiserParams):
        if debug: print("denoiser_callback",params.sampling_step, params.text_cond.shape)
        if self.active:
            if self.x is None: self.x = params.x.shape
            if self.x != params.x.shape: self.hr = True

            # the refiner and the hires pass may load a different checkpoint in the
            # middle of a generation, the new model has to be hooked as well (#64)
            if getattr(self, "handle", None) is not None:
                hook_target(self, current_diffusion_model())

            self.latenti = 0 

            condslist = []
            tokenslist = []
            conds = self.hr_conds_all if self.hrp and self.hr else  self.conds_all
            if conds is not None:
                for step, regions in conds[0]:
                    #print(" ", step,params.sampling_step)
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
                
            global pn, count
            #pn = False if forge or reforge or classic else True
            pn = True
            count = 0
        
        if self.modeltype == "flux" and self.conds:
            self.orig_tokens = params.text_cond[COND_KEY_C].shape[1]
            params.text_cond[COND_KEY_C] = torch.cat([params.text_cond[COND_KEY_C],self.conds[0]],1)

        if self.modeltype == "ZImage" and self.conds:
            self.orig_tokens = params.text_cond.shape[1] - 5
            params.text_cond = torch.cat([params.text_cond[:,:-5,:],self.conds[0],params.text_cond[:,-5:,:]],1)

        if self.modeltype in ("Anima", "Krea") and self.conds:
            text_cond = params.text_cond
            # Anima: (batch, 1, tokens, dim) / Krea2: (batch, tokens, layers, dim)
            axis = 1 if self.modeltype == "Krea" else text_cond.dim() - 2
            addcond = self.conds[0].to(text_cond)
            while addcond.dim() < text_cond.dim():
                addcond = addcond.unsqueeze(1)
            self.orig_tokens = text_cond.shape[axis]
            params.text_cond = torch.cat([text_cond, addcond], axis)
            if debug_arch: print("NegPiP/arch:", self.modeltype, tuple(text_cond.shape), "->", tuple(params.text_cond.shape), "orig_tokens", self.orig_tokens)
            
from pprint import pprint

def current_diffusion_model(sd_model=None):
    """the module NegPiP has to hook for the model that is loaded right now"""
    sd_model = shared.sd_model if sd_model is None else sd_model
    if sd_model is None:
        return None
    if forge:
        return sd_model.forge_objects.unet.model
    return getattr(getattr(sd_model, "model", None), "diffusion_model", None)

def hook_model(self, module, remove=False):
    if module is None:
        return
    if self.modeltype == "flux":
        hook_forwards_f(self, module, remove=remove)
    elif self.modeltype == "ZImage":
        hook_forwards_z(self, module, remove=remove)
    elif self.modeltype == "Anima":
        hook_forwards_a(self, module, remove=remove)
    elif self.modeltype == "Krea":
        hook_forwards_k(self, module, remove=remove)
    else:
        hook_forwards(self, module, remove=remove)

def hook_target(self, module):
    hooked = getattr(self, "hooked_modules", None)
    if module is None or hooked is None:
        return
    if any(module is already for already in hooked):
        return
    if debug_arch and hooked: print("NegPiP: the model was swapped, hooking the new one")
    hook_model(self, module)
    hooked.append(module)

def unload(self,p):
    if hasattr(self,"handle"):
        for module in getattr(self, "hooked_modules", []):
            hook_model(self, module, remove=True)
        self.hooked_modules = []
        del self.handle

# helper functions from LDM
def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def hook_forward(self, module):
    def forward(x, context=None, mask=None, value=None, additional_tokens=None, *args, **kwargs):
        if debug: print(" x.shape:",x.shape,"context.shape:",context.shape,"self.contokens",self.contokens,"self.untokens",self.untokens)
        
        def sub_forward(x, context, mask, additional_tokens, conds,contokens,unconds,untokens, latent = None):
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
                
                xp = main_forward(self, module, ixp,contp,value,mask,additional_tokens,contokens,args,kwargs)
                xn = main_forward(self, module, ixn,contn,value,mask,additional_tokens,untokens,args,kwargs)
            
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

                out = main_forward(self, module, x,context,value,mask,additional_tokens,tokens,args,kwargs)
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
                out = main_forward(self, module, x,context,value,mask,additional_tokens,tokens,args,kwargs)
                return out

        if self.enable_rp_latent:
            if len(self.conds) - 1 >= self.latenti:
                out = sub_forward(x, context, mask, additional_tokens, self.conds[self.latenti],self.contokens[self.latenti],None,None ,latent = True)
                self.latenti += 1
            else:
                out = sub_forward(x, context, mask, additional_tokens, None,None,self.unconds[0],self.untokens[0], latent = False)
                self.latenti = 0
            return out
        else:
            if self.conds is not None and self.unconds is not None and len(self.conds) > 0 and len(self.unconds) > 0:
                return sub_forward(x, context, mask, additional_tokens, self.conds[0],self.contokens[0],self.unconds[0],self.untokens[0])
            else:
                return sub_forward(x, context, mask, additional_tokens, None,None,None,None)
    
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

def main_forward2(self, module, x, context, mask, additional_tokens, tokens, args, kwargs):
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

def main_forward(self, attn, x, context, value = None ,mask = None, temb = None, tokens = [], args = None, kwargs = None):
        q = attn.to_q(x)
        context = context.to(x.dtype)
        context = default(context, x)
        k = attn.to_k(context)
        if value is not None:
            v = attn.to_v(value)
            del value
        else:
            v = attn.to_v(context)

        if self.active:
            if tokens:
                #print(tokens, v.shape)
                #print("v.shape:",v.shape,"start:",tokens+1)
                v[:,-tokens:,:] = -v[:,-tokens:,:] 

        out = attention_function(q, k, v, attn.heads, mask)
        return attn.to_out(out)
    

def attention_function(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False):
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
            (q, k, v),
        )

    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    out = (
        out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    )
    return out

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


def negpip_range(hook_self):
    """position of the NegPiP tokens inside the (already extended) conditioning"""
    if not hook_self.active or not hook_self.contokens or not hook_self.conds:
        return None
    start = getattr(hook_self, "orig_tokens", None)
    if start is None:
        return None
    return start, start + hook_self.conds[0].shape[1]

def negpip_strength(hook_self, length, device, dtype):
    """Per token multiplier for the NegPiP tokens.

    On LLM text encoders the conditioning is built with a neutral weight, so the
    weight from the prompt is applied here: the value vectors are multiplied by the
    (negative) weight instead of only being flipped. On the CLIP based encoders the
    weight is carried by the emphasis of the conditioning itself and the value is
    flipped, which is the original NegPiP behaviour.
    """
    if hook_self.modeltype not in LLM_MODELS:
        return None
    strength = getattr(hook_self, "strength", None)
    if not strength or len(strength) != length:
        return None
    gain = MODEL_GAIN.get(hook_self.modeltype, 1.0)
    return torch.tensor(strength, device=device, dtype=dtype).view(1, -1, 1) * gain

def hook_value_projection(self, module, attr, remove=False, value_dim=None, name=""):
    """Wrap the value projection of an attention module.

    NegPiP only needs the value vectors of its own tokens flipped. Wrapping the
    ``v`` linear instead of reimplementing ``forward`` keeps the extension
    independent of how each architecture implements attention, which changes
    often on Forge Neo. ``value_dim`` is used when q, k and v share one linear.
    """
    linear = getattr(module, attr, None)
    if linear is None:
        return

    if remove:
        original = getattr(linear, "negpip_forward", None)
        if original is not None:
            linear.forward = original
            del linear.negpip_forward
        return

    if hasattr(linear, "negpip_forward"):
        return

    original = linear.forward
    linear.negpip_forward = original
    hook_self = self
    if debug_arch: print("NegPiP/hook:", name or module.__class__.__name__, attr)

    def forward(x, *args, **kwargs):
        out = original(x, *args, **kwargs)
        span = negpip_range(hook_self)
        if span is None:
            return out
        start, end = span
        if end > out.shape[-2]:
            return out
        target = out[..., start:end, :] if value_dim is None else out[..., start:end, -value_dim:]
        strength = negpip_strength(hook_self, end - start, out.device, out.dtype)
        target = target * strength if strength is not None else -target
        if value_dim is None:
            out[..., start:end, :] = target
        else:
            out[..., start:end, -value_dim:] = target
        return out

    linear.forward = forward

def hook_forwards_z(self, root_module: torch.nn.Module, remove=False):
    # Z-Image (NextDiT): caption tokens come first in the joint sequence, q, k and v
    # share a single linear so only the trailing value slice is flipped.
    # Only the joint ``layers`` are hooked. Flipping inside ``context_refiner`` as well
    # subtracts the token from its neighbours before the joint attention subtracts it
    # again, which makes the response non monotonic and reverses it past about -2.
    # ``noise_refiner`` only sees image tokens and must be left alone.
    for name, module in root_module.named_modules():
        if "layers" in name and module.__class__.__name__ == "JointAttention":
            hook_value_projection(self, module, "qkv", remove, value_dim=module.n_local_kv_heads * module.head_dim, name=name)

def hook_forwards_a(self, root_module: torch.nn.Module, remove=False):
    # Anima: classic cross attention, the NegPiP tokens are appended to the context
    for name, module in root_module.named_modules():
        if name.endswith("cross_attn") and module.__class__.__name__ == "SelfCrossAttention":
            hook_value_projection(self, module, "v_proj", remove)

def hook_forwards_k(self, root_module: torch.nn.Module, remove=False):
    # Krea2: single stream DiT, text tokens come first in the joint sequence.
    # ``txtfusion.refiner_blocks`` attends over the text tokens and mixes the NegPiP
    # tokens into the prompt, so it is flipped too. ``txtfusion.layerwise_blocks``
    # attends over the 12 encoder layers instead of tokens and must be left alone.
    for name, module in root_module.named_modules():
        if not name.endswith(".attn") or module.__class__.__name__ != "Attention":
            continue
        if (".blocks." in name and "txtfusion" not in name) or "refiner_blocks." in name:
            hook_value_projection(self, module, "wv", remove, name=name)


class InputAccordionImpl(gr.Checkbox):
    webui_do_not_create_gradio_pyi_thank_you = True
    # the element id has to be unique across every extension that ships this
    # accordion. A shared counter with a per extension offset collides as soon as
    # the tab is built more than once (txt2img, img2img, ...), and the javascript
    # then wires the checkbox of one extension to the accordion of another one
    global_index = 0

    @wraps(gr.Checkbox.__init__)
    def __init__(self, value=None, setup=False, **kwargs):
        if not setup:
            super().__init__(value=value, **kwargs)
            return

        self.accordion_id = kwargs.get('elem_id')
        if self.accordion_id is None:
            self.accordion_id = f"input-accordion-m-negpip-{InputAccordionImpl.global_index}"
            InputAccordionImpl.global_index += 1

        kwargs_checkbox = {
            **kwargs,
            "elem_id": f"{self.accordion_id}-checkbox",
            "visible": False,
        }
        super().__init__(value=value, **kwargs_checkbox)
        self.change(fn=None, _js='function(checked){ inputAccordionChecked("' + self.accordion_id + '", checked); }', inputs=[self])

        kwargs_accordion = {
            **kwargs,
            "elem_id": self.accordion_id,
            "label": kwargs.get('label', 'Accordion'),
            "elem_classes": ['input-accordion-m'],
            "open": False,
        }

        self.accordion = gr.Accordion(**kwargs_accordion)

    def extra(self):
        return gr.Column(elem_id=self.accordion_id + '-extra', elem_classes='input-accordion-extra', min_width=0)

    def __enter__(self):
        self.accordion.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.accordion.__exit__(exc_type, exc_val, exc_tb)

    def get_block_name(self):
        return "checkbox"

def InputAccordion(value=None, **kwargs):
    return InputAccordionImpl(value=value, setup=True, **kwargs)

# Check for Gradio version 4; see Forge architecture rework
IS_GRADIO_4 = version.parse(gr.__version__) >= version.parse("4.0.0")
# check if Forge or auto1111 pure; extremely hacky

# Forge patches

# See discussion at, class versus instance __module__
# https://github.com/LEv145/--sd-webui-ar-plus/issues/24
# Hack for Forge with Gradio 4.0; see `get_component_class_id` in `venv/lib/site-packages/gradio/components/base.py`
if IS_GRADIO_4:
    InputAccordionImpl.__module__ = "modules.ui_components"