import gradio as gr
import torch
import ldm.modules.attention as atm
import modules.ui
import modules
from modules import prompt_parser
from modules import shared

debug = False

class Script(modules.scripts.Script):   
    def __init__(self):
        self.active = False
        self.conds = None
        self.unconds = None
        self.conlen = []   
        self.unlen = []
        self.contokens = []   
        self.untokens = []
        
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
        self.batch = p.batch_size
        self.isxl = hasattr(shared.sd_model,"conditioner")
        
        self.rev = p.sampler_name in ["DDIM", "PLMS", "UniPC"]

        parsed_p = prompt_parser.parse_prompt_attention(p.prompts[0])
        parsed_np = prompt_parser.parse_prompt_attention(p.negative_prompts[0])

        if debug: print(parsed_p)
        if debug:print(parsed_np)

        np ,pn, tp, tnp = [], [], [], []

        for text,weight in parsed_p:
            if text == "BREAK": continue
            if weight < 0:
                np.append([text,weight])
            else:
                tp.append(f"({text}:{weight})" if weight != 1.0 else text)

        tnp =[]
        for text,weight in parsed_np:
            if text == "BREAK": continue
            if weight < 0:
                pn.append([text,weight])
            else:
                tnp.append(f"({text}:{weight})" if weight != 1.0 else text)

        tokenizer = shared.sd_model.conditioner.embedders[0].tokenize_line if self.isxl else shared.sd_model.cond_stage_model.tokenize_line

        p.prompts = [" ".join(tp)]*self.batch
        p.negative_prompts =  [" ".join(tnp)]*self.batch

        p.hr_prompts = p.prompts
        p.hr_negative_prompts = p.negative_prompts

        def conddealer(targets):
            conds =[]
            start = None
            end = None
            for target in targets:
                input = prompt_parser.SdConditioning([f"({target[0]}:{-target[1]})"], width=p.width, height=p.height)
                cond = prompt_parser.get_learned_conditioning(shared.sd_model,input,0)
                if start is None: start = cond[0][0].cond[0:1,:] if not self.isxl else cond[0][0].cond["crossattn"][0:1,:]
                if end is None: end = cond[0][0].cond[-1:,:] if not self.isxl else cond[0][0].cond["crossattn"][-1:,:]
                _, tokenlen = tokenizer(target[0])
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

        if np:
            self.conds, self.contokens = conddealer(np)
        
        if pn:
            self.unconds, self.untokens = conddealer(pn)

        resetpcache(p)
        
        def calcsets(A, B):
            return A // B if A % B == 0 else A // B + 1

        self.conlen = calcsets(tokenizer(p.prompts [0])[1],75)
        self.unlen = calcsets(tokenizer(p.negative_prompts[0])[1],75)

        self.handle = hook_forwards(self, p.sd_model.model.diffusion_model)

        print(f"NegPiP enable: Pos:{self.contokens}, Neg{self.untokens}")

    def postprocess(self, p, processed, *args):
        if hasattr(self,"handle"):
            hook_forwards(self, p.sd_model.model.diffusion_model, remove=True)
            del self.handle

def hook_forward(self, module):
    def forward(x, context=None, mask=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        if debug: print(x.shape,context.shape)
        def main_foward(x, context, mask, additional_tokens, n_times_crossframe_attn_in_self, tokens):
            h = module.heads

            q = module.to_q(x)

            context = atm.default(context, x)
            k = module.to_k(context)
            v = module.to_v(context)
            if debug: print(h,context.shape,q.shape,k.shape,v.shape)
            q, k, v = map(lambda t: atm.rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
            sim = atm.einsum('b i d, b j d -> b i j', q, k) * module.scale

            if self.active:
                for token in tokens:
                    start = (v.shape[1]//77 - len(tokens)) * 77
                    if debug: print(start+1,start+token)
                    v[:,start+1:start+token,:] = -v[:,start+1:start+token,:] 
                    start = start + 77

            if atm.exists(mask):
                mask = atm.rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = atm.repeat(mask, 'b j -> (b h) () j', h=h)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)
            #print(h,context.shape,q.shape,k.shape,v.shape,attn.shape)
            out = atm.einsum('b i j, b j d -> b i d', attn, v)

            out = atm.rearrange(out, '(b h) n d -> b n (h d)', h=h)
            return module.to_out(out)

        if debug: print(x.shape[0],self.batch *2)

        if x.shape[0] == self.batch *2:
            if self.rev:
                contn,contp = context.chunk(2)
                ixn,ixp = x.chunk(2)
            else:
                contp,contn =  context.chunk(2)
                ixp,ixn = x.chunk(2)#x[0:self.batch,:,:],x[self.batch:,:,:]
            
            if self.conds is not None:contp = torch.cat((contp,self.conds),1)
            if self.unconds is not None:contn =  torch.cat((contn,self.unconds),1)
            xp = main_foward(ixp,contp,mask,additional_tokens,n_times_crossframe_attn_in_self,self.contokens)
            xn = main_foward(ixn,contn,mask,additional_tokens,n_times_crossframe_attn_in_self,self.untokens)
        
            out = torch.cat([xn,xp]) if self.rev else torch.cat([xp,xn])

        else:
            if debug: print(context.shape[1] , self.conlen,self.unlen)
            tokens = []
            if context.shape[1] == self.conlen * 77:
                if self.conds is not None:
                    context = torch.cat([context,self.conds],1)
                    tokens = self.contokens
            elif context.shape[1] == self.unlen * 77:
                if self.unconds is not None:
                    context = torch.cat([context,self.unconds],1)
                    tokens = self.untokens
            out = main_foward(x,context,mask,additional_tokens,n_times_crossframe_attn_in_self,tokens)
        return out

    return forward

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