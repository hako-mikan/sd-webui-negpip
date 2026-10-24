# NegPiP - Negative Prompt in Prompt
[<img src="https://img.shields.io/badge/lang-Egnlish-blue.svg?style=plastic" height="25" />](README.md)
[<img src="https://img.shields.io/badge/言語-日本語-green.svg?style=plastic" height="25" />](README_jp.md)
[<img src="https://img.shields.io/badge/语言-中文-red.svg?style=plastic" height="25" />](README_cn.md)
[<img src="https://img.shields.io/badge/Support-%E2%99%A5-magenta.svg?logo=github&style=plastic" height="25" />](https://github.com/sponsors/hako-mikan)


負の効果を持つプロンプトを使えるようになります  

## 更新情報 2026.10.25.0100(JST)
- Forge NEOのMOSS-Imageに対応しました
- 最近のForge NEOのtext encoderで`'Qwen06Engine' object has no attribute 'tokenize_line'`のエラーが出る問題を修正しました(#73, #74)
- Animaでバッチサイズを2以上にするとエラーになる問題を修正しました(#71)
- CFGが1より大きいとき、Forge NEOがプロンプトとネガティブプロンプトをまとめて計算した場合にネガティブプロンプト側まで影響を受けていた問題を修正しました
- Forge NEOのtext encoder更新以降、Z-ImageとKrea2で負の単語が`(word:1.0)`の括弧ごと文字列としてエンコードされていた問題を修正しました

# 概要
この拡張は、stable diffusion web-uiのプロンプトおよびクロスアテンションを拡張して、負の効果を持つプロンプトをプロンプト内で、正の効果を持つプロンプトをネガティブプロンプト内で使用できるようにします。通常、描きたくないものはネガティブプロンプトに書かれますが、ネガティブプロンプトの計算上、あまり効果が現れないことがあります。この拡張では、プロンプトと同程度の効果を持つ負のプロンプトを使用できるようにします。これにより、以前はネガティブプロンプトに置いて値を大きくしすぎて崩壊していたような単語でも、プロンプトに負の効果を持たせることができ、より強い効果が期待できます。

# 使い方
Activeにチェックを入れることで有効になります。プロンプト入力画面において (word:-1)のようにマイナスの値を入れることで負の効果を持つようになります。ネガティブプロンプトでも有効で、この場合は正の効果を持ちます。

### 重みについて
効果が現れないときは値を大きくしてください。重みはスイッチではなく強度なので、必要な大きさはモデルによって大きく変わります。1より小さい値ではほとんど何も起きないことが多く、**最近のXLモデルでは2以上にしないと効果が現れない場合があります**。`-1` → `-1.5` → `-2` → `-3` のように段階的に上げて、対象が消えるところを探してください。対象が消えるより先に画像が破綻する場合は、その単語がその要素を出している単語ではない可能性が高いので、実際にその概念を担っている単語を指定してみてください。

　**Krea2では重みが効かず**、符号だけが効きます。`-0.01`と`-10`で同じ画像になります。Krea2はsingle streamのモデルなので、NegPiPのトークンは画像トークンと同じ残差ストリームに乗り、各ブロックの先頭でそのストリーム全体にRMS normがかかります。value側に与えたスケールは次のブロックに届く前に正規化で消えてしまい、向きだけが残ります。Krea2ではスイッチとして使ってください。

　**MOSS-Image**では重みは強度として働き、CLIP系のモデルと同じくらいの尺度です。`-1`程度で対象が消え、`-3`あたりからはその要素だけでなく画像全体が変わり始めます。

これはgothic dressというプロンプトで作りました。ネガティブプロンプトに`(black:1.8)`と入れているにもかかわらず黒いですね。gothicの黒を消しきれないです。  
![image1](https://github.com/hako-mikan/sd-webui-negpip/blob/imgs/sample.jpg)

これはNegPiPでプロンプトに`(black:-1.8)`を入れました。黒くなくなりましたね。
![image2](https://github.com/hako-mikan/sd-webui-negpip/blob/imgs/sample2.jpg)

ちなみに、NegPiPもネガティブプロンプトも使わないとこうなります。
![image3](https://github.com/hako-mikan/sd-webui-negpip/blob/imgs/sample3.jpg)

## マジカルダンディ
マジカルダンディはマジカルなダンディです。マジカルなダンディを呼び出すことはとても難しいです。それはmagical girlとdandyを共存させる必要があるからです。でもダンディは弱いです。girlは強いです。とても強いです。なのでダンディは負けてしまいます。ネガティブプロンプトに`(girl:1.8)`って入れても出てきません。
![](https://github.com/hako-mikan/sd-webui-negpip/blob/imgs/sample4.jpg)

なのでプロンプトの方に`(girl:-1.6)`と入れてgirlを消す必要があるんじゃんよ。
![](https://github.com/hako-mikan/sd-webui-negpip/blob/imgs/sample5.jpg)

## Txt2Img/Img2Imgタブで拡張を表示しない
Web-UIで、Settings > NegPiPに移動します。  
"Hide in Hide in Txt2Img/Img2Img tab"のオプションをチェックしてください。  
これをチェックすると、Settingsの"Active"が有効になります。

## APIで有効化する
スクリプトの指定を以下のように記述してください。

```
"alwayson_scripts": {
	"NegPiP": {
		"args": [True]
}}
```

## ADetailerとの併用について
Web-UIで、Settings > ADetailerに移動してください。  
「Script names to apply to ADetailer (separated by comma)」と書かれたテキストボックスの末尾に「,negpip」を追加し、Apply Settings  


