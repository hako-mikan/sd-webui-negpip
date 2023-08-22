# NegPiP - Negative Prompt in Prompt
[English](README.md) | [日本語](README_jp.md) | 中文

在 SD WebUI 中允许使用反向咒语（提示词）

# 摘要
该扩展增强了 SD WebUI 的 提示 和 交叉注意力 功能，允许在常规咒语中使用反向咒语，在反向咒语中使用正向咒语。通常情况下，不需要的元素会被放置在反向咒语中，但反向咒语在计算中不一定会产生重大影响。有了这一扩展，就可以使用效果与常规咒语相当的反向咒语。通过在咒语中加入反向效果，即使是以前在反向咒语中数值增加过多而可能崩溃的咒语，也能获得更强的效果。

# 使用
选中“Active”复选框后，该插件将生效。在提示词输入框中，输入负值（如`(word:-1)`）时会产生反向作用。它也适用于反向咒语，在这种情况下，它将产生正向效应。

这是根据“gothic dress”的提示创建的。尽管在否定提示中包含了`(black:1.8)`，但它仍然是黑色的。要完全消除`gothic`一词的黑色似乎是不可能的。

![image1](https://github.com/hako-mikan/sd-webui-negpip/blob/imgs/sample.jpg)

下图在 NegPiP 中使用`(black:-1.8)`创建，不再是黑色。

![image2](https://github.com/hako-mikan/sd-webui-negpip/blob/imgs/sample2.jpg)

顺带一提，这是不使用 NegPiP 或负面提示时的结果。
![image2](https://github.com/hako-mikan/sd-webui-negpip/blob/imgs/sample3.jpg)

## 魔法花公
魔法花公就是带有花花公子属性的魔法少男，但是想要召唤他异常困难。这是因为它需要魔法少女（magical girl）和花花公子（dandy）共存。但dandy属性却很弱。girl属性很坚强。非常强。所以，dandy最终还是输了。即使在反向中输入`(girl:1.8)`，花花公子也还算是不会出现。
![](https://github.com/hako-mikan/sd-webui-negpip/blob/imgs/sample4.jpg)

因此，可能有必要在正向咒语中输入`(girl:-1.6)`来削弱 girl。
![](https://github.com/hako-mikan/sd-webui-negpip/blob/imgs/sample5.jpg)
