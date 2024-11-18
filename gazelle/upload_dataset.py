import datasets
from datasets import load_dataset

dsn = "parler-tts/mls_eng_10k"

ds = load_dataset(dsn)

ds1 = ds["train"].select(range(200000))
ds2 = ds["train"].select(range(0, 500000))

ds_dev = ds["dev"].select(range(3000))

ds1.push_to_hub("amuvarma/mls-eng-10k-200k")
ds2.push_to_hub("amuvarma/mls-eng-10k-500k")

ds_dev.push_to_hub("amuvarma/mls-eng-10k-dev-3k")
