import gradio as gr
import torch
import ldm.modules.attention as atm
import modules.ui
import modules
from modules import prompt_parser
from modules import shared

class Script(modules.scripts.Script):   
    def __init__(self):
        self.active = False
        self.np = []   
        self.pn = []
        self.npt = []   
        self.pnt = []
        
    def title(self):
        return "NegPiP"

    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible

    def ui(self, is_img2img):      
        with gr.Accordion("NegPiP", open=False):
            active = gr.Checkbox(value=False, label="active",interactive=True,elem_id="cdt-disable")

        return [active]

    def process_batch(self, p, active,**kwargs):
        self.__init__()
        if not active:return

        self.active = active
        self.isxl = hasattr(shared.sd_model,"conditioner")

        parsed_p = prompt_parser.parse_prompt_attention(p.prompts[0])
        parsed_np = prompt_parser.parse_prompt_attention(p.negative_prompts[0])

        for text,weight in parsed_p:
            if weight < 0:
                self.np.append(text)
                p.prompts = [x.replace(f"{text}:{weight}",f"{text}:{-weight}") for x in p.prompts]

        for text,weight in parsed_np:
            if weight < 0:
                self.pn.append(text)
                p.negative_prompts = [x.replace(f"{text}:{weight}",f"{text}:{-weight}") for x in p.negative_prompts]

        tokenizer = shared.sd_model.conditioner.embedders[0].tokenize_line if self.isxl else shared.sd_model.cond_stage_model.tokenize_line

        ptokens, ptokensum = tokenizer(p.prompts[0])
        for target in self.np:
            ttokens, _ = tokenizer(target)
            i = 1
            while ttokens[0].tokens[i] != 49407:
                for (j, maintok) in enumerate(ptokens): 
                    if ttokens[0].tokens[i] in maintok.tokens:
                        self.npt.append(maintok.tokens.index(ttokens[0].tokens[i]) + 75 * j)
                i += 1

        ntokens, ntokensum = tokenizer(p.negative_prompts[0])
        for target in self.pn:
            ttokens, _ = tokenizer(target)
            i = 1
            while ttokens[0].tokens[i] != 49407:
                for (j, maintok) in enumerate(ntokens): 
                    if ttokens[0].tokens[i] in maintok.tokens:
                        self.pnt.append(maintok.tokens.index(ttokens[0].tokens[i]) + 75 * j)
                i += 1

        self.pset = ptokensum // 75 + 1
        self.nset = ntokensum // 75 + 1

        self.eq = self.pset == self.nset

        self.handle = hook_forwards(self, p.sd_model.model.diffusion_model)

    def postprocess(self, p, processed, *args):
        if hasattr(self,"handle"):
            hook_forwards(self, p.sd_model.model.diffusion_model, remove=True)
            del self.handle

def hook_forward(self, module):

    def forward(x, context=None, mask=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        h = module.heads

        q = module.to_q(x)
        context = atm.default(context, x)
        k = module.to_k(context)
        v = module.to_v(context)
        q, k, v = map(lambda t: atm.rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = atm.einsum('b i d, b j d -> b i j', q, k) * module.scale

        if self.active :
            if self.eq:
                for t in self.npt:
                    v[0:v.shape[0]//2,t,:] = -v[0:v.shape[0]//2,t,:]
                for t in self.pnt:
                    v[v.shape[0]//2:,t,:] = -v[v.shape[0]//2:,t,:]
            else:
                if v.shape[0] == self.pset * 77:
                    for t in self.npt:
                        v[:,t,:] = -v[:,t,:]
                elif v.shape[0] == self.nset * 77:
                    for t in self.pnt:
                        v[:,t,:] = -v[:,t,:]

        if atm.exists(mask):
            mask = atm.rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = atm.repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)

        out = atm.einsum('b i j, b j d -> b i d', attn, v)

        out = atm.rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return module.to_out(out)

    return forward

def hook_forwards(self, root_module: torch.nn.Module, remove=False):
    for name, module in root_module.named_modules():
        if "attn2" in name and module.__class__.__name__ == "CrossAttention":
            module.forward = hook_forward(self, module)
            if remove:
                del module.forward