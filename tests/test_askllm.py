# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from logging import DEBUG, StreamHandler, getLogger

# from datasets import load_dataset
from transformers import (  # noqa: F401
    AutoModelForCausalLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

import nano_askllm

# Set logging level to DEBUG.
logger = getLogger("nano_askllm.askllm")
logger.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.addHandler(handler)


def test_version():
    assert nano_askllm.__version__ == "0.1.0"
    print("test_version passed")


def test_askllm():
    # load the model and tokenizer
    model_id = "google/gemma-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    # model_id = "google/flan-t5-large"
    # tokenizer = T5Tokenizer.from_pretrained(model_id)
    # model = T5ForConditionalGeneration.from_pretrained(model_id)

    # Load 2 English and 2 Japanese datapoints from the C4 dataset like this:
    #
    # c4_en_train = load_dataset("allenai/c4", "en", split="train", streaming=True)
    # datapoints_en = [item["text"] for item in list(c4_en_train.take(2))]
    # c4_ja_train = load_dataset("allenai/c4", "ja", split="train", streaming=True)
    # datapoints_ja = [item["text"] for item in list(c4_ja_train.take(2))]
    # datapoints = datapoints_en + datapoints_ja
    #
    # The datapoints should be like this:
    #
    datapoints = [
        "Beginners BBQ Class Taking Place in Missoula!\nDo you want to get better at making delicious BBQ? You will have the opportunity, put this on your calendar now. Thursday, September 22nd join World Class BBQ Champion, Tony Balay from Lonestar Smoke Rangers. He will be teaching a beginner level class for everyone who wants to get better with their culinary skills.\nHe will teach you everything you need to know to compete in a KCBS BBQ competition, including techniques, recipes, timelines, meat selection and trimming, plus smoker and fire information.\nThe cost to be in the class is $35 per person, and for spectators it is free. Included in the cost will be either a t-shirt or apron and you will be tasting samples of each meat that is prepared.",  # noqa: E501  # cspell:disable-line
        "Discussion in 'Mac OS X Lion (10.7)' started by axboi87, Jan 20, 2012.\nI've got a 500gb internal drive and a 240gb SSD.\nWhen trying to restore using disk utility i'm given the error \"Not enough space on disk ____ to restore\"\nBut I shouldn't have to do that!!!\nAny ideas or workarounds before resorting to the above?\nUse Carbon Copy Cloner to copy one drive to the other. I've done this several times going from larger HDD to smaller SSD and I wound up with a bootable SSD drive. One step you have to remember not to skip is to use Disk Utility to partition the SSD as GUID partition scheme HFS+ before doing the clone. If it came Apple Partition Scheme, even if you let CCC do the clone, the resulting drive won't be bootable. CCC usually works in \"file mode\" and it can easily copy a larger drive (that's mostly empty) onto a smaller drive. If you tell CCC to clone a drive you did NOT boot from, it can work in block copy mode where the destination drive must be the same size or larger than the drive you are cloning from (if I recall).\nI've actually done this somehow on Disk Utility several times (booting from a different drive (or even the dvd) so not running disk utility from the drive your cloning) and had it work just fine from larger to smaller bootable clone. Definitely format the drive cloning to first, as bootable Apple etc..\nThanks for pointing this out. My only experience using DU to go larger to smaller was when I was trying to make a Lion install stick and I was unable to restore InstallESD.dmg to a 4 GB USB stick but of course the reason that wouldn't fit is there was slightly more than 4 GB of data.",  # noqa: E501  # cspell:disable-line
        "生八つ橋のタグまとめ | エキサイトブログ\n生八つ橋のタグまとめ\n「生八つ橋」のタグがついている新着記事と人気記事をまとめました。エキサイトブログには生八つ橋に関連するブログ（日記、記録、写真、レビュー、噂、まとめ）がたくさん投稿されています。\n「生八つ橋」タグの記事（4）\n生八つ橋いろいろ\n京都旅行のお土産(我が家用)に色々な生八つ橋を買ってきました。我が家はみんな八つ橋ファンなのです～。地元の方達や京都を案内してくれたYさんは「八つ橋なんてもう何年も食べたことないわ～～～」と仰っていましたが、いやいや、美味しいですよ～～！！私は大好きです！まぁ確かに私は東京出身ですが、雷おこしや人形焼きは食べませんからね～。それと同じことでしょうか＾＾；とはいえ、舟和の芋ようかんや東京ばな奈...\n2020/03/06 23:54 - ＮＹの小さな灯り\u3000～ヘアメイク日記～\n黒糖きな粉\n冬限定の生八つ橋「ふゆおたべ」\n日本から戻る時に大抵空港で生八つ橋か信玄餅のどちらかを必ず買って帰ってくるのですが、今回は珍しいバージョンを見かけたので生八つ橋を買う事にしました。それがこちら、冬限定の生八つ橋「ふゆおたべ」です。切り絵のデザインも素敵ですよね＾＾お味は黒豆と栗きんとんの二種類。おせち料理みたいです～♪栗きんとんは元々味がそんなに個性的でない事もあり、あまりよく分からなかったのですが（ふんわり甘くて美味しい...\n2019/03/06 00:30 - ＮＹの小さな灯り\u3000～ヘアメイク日記～\n【株式会社 美十】ショコラこたべ\n日帰り京都旅行、あまり時間がなかったのでお土産はすべて京都駅周辺で済ませてしまいました。が、数年ぶりに訪れた京都ではお土産にも大きな変化が…！代表的な京都土産のひとつ、生八つ橋ってあのニッキの味わいがなんか苦手…っていう方も多いと思うんですけど、今はそういう定番以外にもイマドキ風な生八つ橋がすごくいっぱいあって驚きました！いろんなメーカーがあると思うんですけど、「おたべ」で知られている株式会...\n2018/01/28 21:07 - 岐阜うまうま日記（旧：池袋うまうま...\n大学生の孫が、部活の全国へ大会で、京都へ４日間行ってきたそうです。おみやげに、おたべとお茶を買ってきました。来年は、福井県だそうです。まぁ！アルバイトをしているからいいけれど、交通費がかかりそう！！私は八十八を見て、てっきり八つ橋かと思って開けて見たらお茶でした。ちゃんと、考えて買ってきたのでした。我が家の飲んでいるお茶は、賄い茶、たくさん飲むので、これで十分です。孫のおみやげのお茶は、細く...\n2016/08/16 17:37 - みすずのつぶやき",  # noqa: E501  # cspell:disable-line
        "廃棄物をチップ／ペレットに - Gneuss\nフィルム、繊維、更には市場回収品であるPETボトルフレークなどのリサイクル可能な廃棄物は、高水準の純度で高品質ペレット/チップに加工し、生産押出機でバージン材料と混ぜることができます。\nStartseite アプリケーション例 Granulat & Regranulat\n繊維、フィルムやダンゴなどのリサイクル可能な廃棄物は、シュレッダーに入れられ、金属セパレーター付コンベアでアジエータ―を搭載した容器に投入されます。そこから、材料はオーガや詰め込みフィーダーによって押出機に運ばれます。\nあるいは、サイズを圧縮し塊となったリサイクル可能な廃棄物は、MRS押出機に直接入れることができます。サイズ圧縮と押し出しの処理工程を切り離すことの長所は、材料の投下（そしてさらにスループット率も）が均一にされ、そして、金属は材料フローから確実に排出されるところにあります。このシステムを統合化システム上で達成することは、一般的に非常に難しいです。\n押出機はゆっくりとポリエステルのゴミを溶かし、MRSの脱気セクションにおいて、表面に付着した水分、吸収された水分、スピン仕上げオイル、その他の添加物、または印刷インク（フィルムリサイクル時）といった、揮発成分を含んだポリマーをきれいにします。真空下の集中的で非常に効率的な表面交換は、優れたパフォーマンスを確実にします。ガラス、紙、またはセラミックのような固体物質は、全自動の、セルフクリーニング回転式スクリーンチェンジャーや、フィルターに通された溶融樹脂からペレタイジングシステム（チップ・カッター）へ移されることによって、取り除かれます。\nオプションとして、（たとえばより高いIV値材料が必要とされるとき）溶融ポリマーは、ペレット化される前に、正確なIV値増加を実現するために、直接IV値ブースターシステムJUMPに投入することができます。\nペレタイジング／チップカッティングシステムは、後に続く結晶化する円筒ペレット向け、あるいは、余熱結晶体の球体のペレットアンダーウォーターダイフェイスカッティングシステム向けの、アンダーウォーターストランドペレタイジングプロセスとして特化されることができます。このように、高品質ペレット/チップは、最大100%リサイクル可能な廃棄物から製造することができます.\nGNEUSS社のペレット／チップ産業向けのリサイクル構想の利点\n幅広い特性を持ったペレットでも対応可能です\nユーザーが求める正確な粘度調整ができ、様々なアプリケーションの条件を満たすことが出来ます\nユーザーがブレンドを調整することが出来ます\n広範囲にわたる様々な種類の添加物を、簡単に溶融物に添加することが出来ます\n本来は廃棄物となるようなリサイクル可能材料から作られる高品質ペレットを販売出来ることによる、新しい市場開拓の実現が出来ます\nファイバー、フィルム、生産スタートアップ時にできる廃棄物のような、容積密度の低い廃棄物は、団子状にまとめられ、押出機に投入できるようなサイズにするために、シュレッダーにかけられます。その原料は、金属探知機付のコンベアベルトによって、アジエータ―付の中間容器へと輸送されます。そこから、容積の大きい原料はドリルもしくは振動フィーダーを用いて押出機に投入されます。押出機内で、原料は溶かされ、脱揮され、さらにコンタミを取り除かれます。水分や油分といった揮発性のコンタミは抽出除去されます。一方で固体のコンタミはロータリーフィルトレーションシステムによって抽出除去されます。ペレタイジングの後で、既に揮発性及び個体のコンタミを取り除かれ浄化されたペレット／チップは、フィルムやシート、ファイバー、梱包テープなどといった産業において、バージン材の代わりに用いることが可能です。\nファイバーや、フィルム、スタートアップ時のロス材料を用います。これらは、サイズは減らされ、そして必要であれば団子状にされ、金属探知機を通ってMRS押出機に投入されます。そこで原料は溶かされ、溶け込んだもしくは表面に付着している水分や、スピン仕上げオイル、その他工程で用いられる油分といった揮発性のコンタミを抽出し脱揮します。固体のコンタミは、ロータリーフィルトレーションシステムおよび、ブースターポンプを用いて抽出されます。溶融PETは液相IVブースターシステムJUMPへ投入されます。JUMPリアクター内の撹拌及びミキシング装置もまた、真空下において、高いポリマー溶融表面交換率によって、設定されたIV値を実現することが出来ます。粘度は、ラインコントロールシステムとの組み合わせにおけるオンライン粘度計を用いて調節することが出来ます。溶融ポリマーはロータリーフィルトレーションシステムを通って、バキュームから送り出されたのち、ペレット化されます。",  # noqa: E501  # cspell:disable-line
    ]
    #
    # You can see the actual content at the following URLs:
    # https://huggingface.co/datasets/allenai/c4/viewer/en?row=0
    # https://huggingface.co/datasets/allenai/c4/viewer/en?row=1
    # https://huggingface.co/datasets/allenai/c4/viewer/ja?row=0
    # https://huggingface.co/datasets/allenai/c4/viewer/ja?row=1

    # Default prompt template is not suitable for gemma-2b-it.
    # I changed "OPTIONS:" format from "\n- yes\n- no\n" to " yes / no".
    # I added "ANSWER:" to the last line to increase the probability of "yes" or "no" being the first token.
    # TODO: prompt engineering is necessary for each model.
    prompt_template = """###
{datapoint}
###

Does the previous paragraph demarcated within ### and ### contain informative signal for pre-training a large-language model? An informative datapoint should be well-formatted, contain some usable knowledge of the world, and strictly NOT have any harmful, racist, sexist, etc. content.

OPTIONS: yes / no
ANSWER:"""  # noqa: E501

    # Typical yes tokens, depending on the model.
    yes_tokens = ["yes", "Yes", "YES", " yes", " Yes", " YES"]  # for gemma-2b-it
    # yes_tokens = ["yes", "Yes"]  # for flan-t5-large

    llm = nano_askllm.AskLLM(tokenizer, model, prompt_template=prompt_template, yes_tokens=yes_tokens)
    assert llm is not None
    prompts = llm.get_prompts(datapoints)
    # print(prompts)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    # assert inputs.input_ids.shape[0] == 4 and inputs.input_ids.shape[1] == 1109
    results = llm.ask(inputs)
    print(results)
    # assert str(results) == "tensor([0.9997, 0.1600, 0.9776, 0.9910])"
    del results, inputs, prompts, llm, model, tokenizer

    print("test_askllm passed")
