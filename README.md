# NegPiP - Negative Prompt in Prompt
English | [日本語](README_jp.md) | [中文](README_cn.md)

Extension for Stable Diffusion web-ui enables negative prompt in prompt

Update 2023.09.05.2000(JST)
- Prompt Edittingに対応
- Regional Prompterに対応(最新版のRegional Prompterが必要)
- 負の値を入れていないときでも有効化したときに生成結果が変わる問題を修正

- Supports Prompt Editing
- Supports Regional Prompter (latest version of Regional Prompter required)
- Fixed the issue where generated results change even when negative values are not entered

- 支持Prompt Editting
- 支持区Regional Prompter(需要最新版Regional Prompter)
- 修复了即使没有输入负值时激活也会改变生成结果的问题


# Summary
This extension enhances the stable diffusion web-ui prompts and cross-attention, allowing for the use of prompts with negative effects within regular prompts and prompts with positive effects within negative prompts. Typically, unwanted elements are placed in negative prompts, but negative prompts may not always have a significant impact in calculations. With this extension, it becomes possible to use negative prompts with effects comparable to regular prompts. This enables stronger effects even for words that might have collapsed when their values were increased too much in negative prompts before, by incorporating negative effects into the prompts.

# Instructions
By checking the "Active" box, it will become effective. In the prompt input screen, entering a negative value like `(word:-1)` will give it a negative effect. It also works with negative prompts, in which case it will have a positive effect.

This was created with the prompt "gothic dress". Despite including `(black:1.8)` in the negative prompt, it's still black. It seems impossible to completely eliminate the blackness of word `gothic`.

![image1](https://github.com/hako-mikan/sd-webui-negpip/blob/imgs/sample.jpg)

Following image created using `(black:-1.8)` in the prompt with NegPiP. It's no longer black.

![image2](https://github.com/hako-mikan/sd-webui-negpip/blob/imgs/sample2.jpg)

By the way, this is what happens when you don't use either NegPiP or negative prompts.
![image2](https://github.com/hako-mikan/sd-webui-negpip/blob/imgs/sample3.jpg)

## Magical Dandy
Magical Dandy is a magical dandy. Summoning a magical dandy is very difficult. That's because it requires coexistence of a magical girl and a dandy. But the dandy is weak. The girl is strong. Very strong. So the dandy ends up losing. Even if you put `(girl:1.8)` in the negative prompt, it won't come up. 
![](https://github.com/hako-mikan/sd-webui-negpip/blob/imgs/sample4.jpg)

Therefore, it may be necessary to input `(girl:-1.6)` in the prompt to remove the girl.
![](https://github.com/hako-mikan/sd-webui-negpip/blob/imgs/sample5.jpg)
