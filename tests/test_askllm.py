# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

from logging import DEBUG, StreamHandler, getLogger
from time import time

import pytest
import torch
from datasets import load_dataset
from transformers import (  # noqa: F401
    AutoModelForCausalLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from nano_askllm import AskLLM, __version__

# Set logging level to DEBUG.
logger = getLogger("nano_askllm.askllm")
logger.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.addHandler(handler)


def test_version():
    assert __version__ == "0.2.3"
    print("test_version passed")


def test_paper_appendix_e():
    model_id = "google/flan-t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_id)
    model = T5ForConditionalGeneration.from_pretrained(model_id, device_map="auto")
    llm = AskLLM(tokenizer, model)
    assert isinstance(llm, AskLLM)

    datapoints = [
        # Appendix E.1. High-quality Samples Identified by ASK-LLM
        "What constitutes overtime for a part-time employee? Question: What is overtime for a part-time employee? Overtime for a part-time employee is time that is beyond the part-time employee’s ordinary hours of work or outside the agreed number of hours of work, as specified in their employment contract.",  # noqa: E501
        "Viva La Vegan! - Can a Vegan Lifestyle Help to Get Rid of Ocean Dead Zones? Can a Vegan Lifestyle Help to Get Rid of Ocean Dead Zones? A dead zone is an area at the bottom of the ocean that is oxygen depleted and cannot maintain any marine life. The biggest cause of these dead zones is an overflow of fertilizers, sewage and industrial pollutants being pumped into rivers all over the world. Thankfully dead zones can be reversed and living a vegan lifestyle can help enormously and I’ll show you how. What are Ocean Dead Zones?\n......\nVegans don’t want to harm the planet.On the contrary they want to save it and what better way than living with nature instead of against it and helping the planet in ways we probably never even realised, like helping to reverse our oceans dead zones. Next time you think about buying something you don’t need, or eating food that is highly processed or non-organic, spare a thought for the largely unknown dead zones and how overconsumption and an unnatural lifestyle is slowly killing both you and them.",  # noqa: E501  # cspell: disable-line
        "Question: Is it necessary to dredge ponds and lakes in the upper coastal region of South Carolina?\nAnswer: It is necessary to dredge ponds and lakes in South Carolina, in the upper coastal region of South Carolina.\nEach lake and each pond is a different environment and as years pass, these environments accumulate a lot of sediment.\nThey tend to fill in with storm water runoff, they tend from natural leafy materials—whether it be grass clippings, leafy materials, storm water fun off, sand, silt, sediment, muck, mire.\nAll of these produce in the bottoms of pond beds and lake beds.\nSo it is absolutely necessary to do an evaluation every so many years to determine whether or not you need to remove the sediment that’s accumulated.",  # noqa: E501
        # Appendix E.2. Low-quality Samples Identified by ASK-LLM
        "Release name : Juiced2.Hot.Import.Nights-Multi5-RELOADED. ? Format : iso Juiced 2: HIN evolves the current street racing scene, letting players experience PC Repack DiRT Rally v1.1 ? Black Box Bears Cant Drift PC torrent uploaded. ? Juiced 2 ? ? ?? ? ???? ???? ? ??? ? ?? ? ? ? ? ????! .\n...\nHIN evolves the current street racing scene, letting players experience the culture of the real-life HIN tour, the nation?s largest lifestyle custom. Juiced 2 Hot Import Nights Torrent. Bittorrent 729.64 MB. Juiced 2 Hot Import Nights Download free torrent at Largest Bittorrent Source with Several Listed Files. Now you can upload screenshots or other images (cover scans, disc scans,...",  # noqa: E501
        "You were a good daughter the first day or two. Now, you are only showing the worst sides of yourself. I can only be sad and disappointed in you.",  # noqa: E501
        "Kids can help you enrich your life? Be a better person? Learn to think about someone else? Apparently whoever said these things has never had children because from everything we have seen and experienced, kids are flat out horrible. College can’t come fast enough.",  # noqa: E501
        # Appendix E.3. Increasing-quality Samples Identified by ASK-LLM
        "The historic city of Manchester now features one of the most interesting public art installations that art lovers have ever witnessed. Design studio, Acrylicize installed five giant lamps in Piccadilly Place that represent the many historic periods that the city has gone through, including; Art Deco, Art Nouveau, Victorian, mid-century, and contemporary. The installation is without any doubt, a great piece of art but unlike other artworks, these are absolutely functional as well. Each lamp provides the many visitors with seating, shelter, light and even heat in the winters. The admirers can also witness the historic stories of Manchester via graphic illustrations on the lamps.",  # noqa: E501  # cspell: disable-line
        "The Cokin Yellow and Pink Center Spot filter has a clear center and diffused yellow and pink edges. Theses diffused edges will be produce blur while leaving the center sharp. The filter effect is directly influenced by the f-stop and the focal length. A lens shot at f/1.4 will see a greater blurring effect than f/8.0 and a 85mm lens will see more blur than a 28mm. Additionally, a longer focal length lens will visually increase the size of the center spot area because it sees less of the filter area.",  # noqa: E501  # cspell: disable-line
        "Provide hoist coverage and 200 degree rotation for individual use in bays, along walls, or columns of plants, or as a supplement to an overhead crane or monorail system. This jib has the advantage of providing maximum lift for the hoist, since it can be installed very close to the underside of the lowest ceiling obstruction. It is composed of a vertical mast mounted to 2 brackets on a wall or vertical building beam with a boom that cantilevers out, perpendicular from the wall at the top.",  # noqa: E501
        # Appendix E.4. Decreasing-quality Samples Identified by ASK-LLM
        "one filled with goodwill and cheer. who have supported me thru the year. I wouldn’t be changing careers. instead of on strange people’s rears. Wishes You a Healthy, Happy Holidays! Ah, how the mighty have fallen! And a Merry fave to you ... and a happy new rear. From one Xmas humor story to another, enjoyed this! Thanks Jack & Susan! Doug, I checked him out–wonderful stuff! Will pass along the good word. Fun and funny–as always! Thanks for the cheer! I can only fave this once, but I’ve looked at it repeatedly over what has been a bizarre week– and each time you’ve given me a laugh. That’s a gift Bob and I’m grateful! Best of holidays to you and a great New Year!",  # noqa: E501
        "I hear people saying that vinyl records have a better sound quality than CDs or even DVDs. A mini LP is a CD version of something that was originally released as a 12\" (12 inch) vinyl LP. In many cases the packaging is superior to, or at least. Vitalogy; Studio album by Pearl Jam; Released: Vinyl: November 22, 1994 CD: December 6, 1994: Recorded: November 1993 – October 1994: Studio: Bad Animals Studio. Browse best sellers, new releases, AutoRip CDs and vinyl records, deals, vinyl Audio CD. 7.99. From A Room: Volume 1. Chris Stapleton. Audio. The one and only CD, DVD, VIDEO, DJ, VINYL, ERO store. Search our full catalog. Recordstore.co.uk. The UK’s leading online record store. Buy new and exclusive signed bundles, CDs, LPs, Merchandise and box sets. Recordstore Day, every. Vinyl Records to CD Conversion - Cheapest on the net! High-quality, standards-compliant CD-Audio of your favorite vinyl records, saved for posterity. Custom CD, DVD Vinyl Packaging You’re just a click away from a gorgeous, retail-ready CD or DVD in professional disc packaging. We also offer a full-range of Vinyl.\n...\nBuy with confidence as the. Mar 4, 2017 Despite the decline in mainstream CD usage, some consumers still have CD recording needs for radio, vinyl and other formats. Here are our. 12 results . You can finally burn your cassettes and vinyl records to CD with Crosley’s Memory Master II CD Recorder. Just play your cassette or record One Nation is back after the Sold Out New Years Eve event with yet another From its esoteric origins releasing field recordings of steam engines on vinyl to our latest critically acclaimed Ultradisc UHR™ SACDs, Mobile Fidelity Sound. How much are worth and valued your rare and collectable vinyl and cd by searching on Music Price Guide archive. Heel veel CD, LP, Vinyl SACD op voorraad, snelle levertijden en altijd superscherp geprijsd en lage verzendkosten, voor 17:00 besteld morgen Some of the greatest music ever made isn t available digitally, on mp3, or on CD; but rather is only available on vinyl. Moreover, if you already have purchased.",  # noqa: E501  # cspell: disable-line
        "A brilliant performance by Year 6 based on The Lion King. Brilliant singing and acting from everyone, congratulations Year 6! A big thank you to all the staff that helped with everything from costumes, set design, make up and directing. A wonderful commemoration of the seven years that Year 6 students have spent at The Good Shepherd. Thank you to all of the parents and staff for attending this celebration and we wish all of the children continued success in their new schools and hope they continue to do themselves proud. Well done to Foundation for showing us what it is to be good friends! This week we have been looking at all the countries in the world that speak Spanish as their native language, there are 21! So throughout school we spent a day learning lots of wonderful things about our chosen country. We looked at maps, flags, famous people, food and so much more! Below is a little glimpse into our fabulous week.\n...\nClick on the links to take a look at some of the brilliant things we got up to! Faith in Families is a charity based here in Nottingham who believe, as we do, that all children have the right to grow up as part of a loving and nurturing family and they provide services for children and families. We learnt lots about adoption and what it can mean for children and their family. We learnt about Fairtrade and all the fantastic work they do around the world. We also discovered lots of products that we did not know were Fairtrade. There was also a sell out Fairtrade food sale, well done everyone! Year 2 have been able to show off our brilliant new high visibility jackets! Now we will be able to stay safe and visible on any out of school trips. We are very lucky to have these donated by Walton & Allen. Thank you! Click on the high visibility jacket to take a look at our super jackets! Year 4 have wowed us with their acting skills in a brilliant performance of Ali Baba - well done Year 4! Year...",  # noqa: E501  # cspell: disable-line
    ]

    scores = llm.ask(datapoints)
    assert isinstance(scores, torch.Tensor) and scores.shape == (len(datapoints),)
    assert all(scores >= 0.0) and all(scores <= 1.0)
    for i, (score, datapoint) in enumerate(zip(scores.tolist(), datapoints)):
        text = datapoint.replace("\n", " ")
        print(f"{i + 1:2d} | {score:.4f} | {text:40.40}")


@pytest.mark.skip()
def test_flan_t5_c4_en():
    # load the model and tokenizer
    # *Flan T5 only works on English datasets.*
    model_id = "google/flan-t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_id)
    model = T5ForConditionalGeneration.from_pretrained(model_id, device_map="auto")

    # Load C4 English dataset.
    # You can see the actual content at the following URLs:
    # https://huggingface.co/datasets/allenai/c4/viewer/en
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

    llm = AskLLM(tokenizer, model)
    assert isinstance(llm, AskLLM)

    batch_size = 2
    num_ask = 5

    print("-" * 80)
    start_time = time()
    for i in range(num_ask):
        print(f"batch {i + 1} start")
        datapoints = [item["text"] for item in list(dataset.take(batch_size))]
        scores = llm.ask(datapoints)
        assert isinstance(scores, torch.Tensor) and scores.shape == (batch_size,)
        assert all(scores >= 0.0) and all(scores <= 1.0)
        for score, datapoint in zip(scores.tolist(), datapoints):
            text = datapoint[:80].replace("\n", " ")
            print(f"score: {score:.4f}\ttext: {text}")
        del scores
        dataset = dataset.skip(batch_size)
        end_time = time()
        print(f"batch {i + 1} end, {(end_time - start_time):.4f} seconds")
        print("-" * 80)
        start_time = end_time

    del llm, dataset, model, tokenizer
    print("test_flan_t5_c4_en passed")


@pytest.mark.skip()
def test_gemma_mc4_ja():
    # load the model and tokenizer
    model_id = "google/gemma-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    # For 4bit quantization on Colab T4 GPU
    # quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    # model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", quantization_config=quantization_config)

    # Load mC4 Japanese dataset.
    # You can see the actual content at the following URLs:
    # https://huggingface.co/datasets/allenai/c4/viewer/ja
    dataset = load_dataset("allenai/c4", "ja", split="train", streaming=True)

    # Default prompt template is not suitable for gemma-2b-it.
    # I changed "OPTIONS:" format from "\n- yes\n- no\n" to " yes/no\n".
    # I added "ANSWER:" to the last line to increase the probability of "yes" or "no" being the first token.
    # TODO: prompt engineering is necessary for each model.
    prompt_template_prefix = "###\n"
    prompt_template_postfix = """
###

Does the previous paragraph demarcated within ### and ### contain informative signal for pre-training a large-language model? An informative datapoint should be well-formatted, contain some usable knowledge of the world, and strictly NOT have any harmful, racist, sexist, etc. content.

OPTIONS: yes/no
ANSWER:"""  # noqa: E501

    yes_tokens = ["yes", "Yes", "YES", " yes", " Yes", " YES"]  # for gemma-2b-it

    llm = AskLLM(
        tokenizer,
        model,
        prompt_template_prefix=prompt_template_prefix,
        prompt_template_postfix=prompt_template_postfix,
        yes_tokens=yes_tokens,
        max_tokens=512,  # You can increase it up to 8192 for gemma-2b-it.
    )
    assert llm is not None

    batch_size = 2
    num_ask = 5

    print("-" * 80)
    start_time = time()
    for i in range(num_ask):
        print(f"batch {i + 1} start")
        datapoints = [item["text"] for item in list(dataset.take(batch_size))]
        scores = llm.ask(datapoints)
        assert isinstance(scores, torch.Tensor) and scores.shape == (batch_size,)
        assert all(scores >= 0.0) and all(scores <= 1.0)
        for score, datapoint in zip(scores.tolist(), datapoints):
            text = datapoint[:80].replace("\n", " ")
            print(f"score: {score:.4f}\ttext: {text}")
        del scores
        dataset = dataset.skip(batch_size)
        end_time = time()
        print(f"batch {i + 1} end, {(end_time - start_time):.4f} seconds")
        print("-" * 80)
        start_time = end_time

    del llm, dataset, model, tokenizer
    print("test_gemma_mc4_ja passed")
