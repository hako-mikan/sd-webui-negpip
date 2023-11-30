# NegPiP - Negative Prompt in Prompt
[<img src="https://img.shields.io/badge/lang-Egnlish-blue.svg?style=plastic" height="25" />](README.md)
[<img src="https://img.shields.io/badge/言語-日本語-green.svg?style=plastic" height="25" />](README_jp.md)
[<img src="https://img.shields.io/badge/语言-中文-red.svg?style=plastic" height="25" />](README_cn.md)
[<img src="https://img.shields.io/badge/Support-%E2%99%A5-magenta.svg?logo=github&style=plastic" height="25" />](https://github.com/sponsors/hako-mikan)


Extension for Stable Diffusion web-ui enables negative prompt in prompt

## Update 2023.10.29.2100(JST)
- Option to hide this extention in t2i/i2i tab [Detail](#hide-this-extention-in-text2imgimg2img-tab),[詳細](README_jp.md#txt2imgimg2imgタブで拡張を表示しない),[解释](README_cn.md#在txt2imgimg2img标签中不显示扩展)

### [For users of ADetailer](#for-users-of-adetailer)/[ADetailerとの併用について](README_jp.md#adetailerとの併用について)/[关于与ADetailer的同时使用](README_cn.md#关于与adetailer的同时使用)

# Overview
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

## Hide this extention in text2img/img2img tab
In the Web-UI, go to Settings > NegPiP.  
Check the "Hide in Hide in Txt2Img/Img2Img tab" option.  
If you check this, the "Active" in Settings will be effective.  

## For users of ADetailer
In the Web-UI, go to Settings > ADetailer.  
Add ",negpip" to the end of the text box labeled "Script names to apply to ADetailer (separated by comma)"  
Click "Apply Settings.  


### Update 2023.09.05.2000(JST)
- Prompt Edittingに対応
- Regional Prompterに対応(最新版のRegional Prompterが必要)
- 負の値を入れていないときでも有効化したときに生成結果が変わる問題を修正

- Supports Prompt Editing
- Supports Regional Prompter (latest version of Regional Prompter required)
- Fixed the issue where generated results change even when negative values are not entered

- 支持Prompt Editting
- 支持区Regional Prompter(需要最新版Regional Prompter)
- 修复了即使没有输入负值时激活也会改变生成结果的问题
