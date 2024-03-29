<hr />
<b>WELCOME TO GP-GPT!</b>
<img src='misc/basically-gp-gpt.jpeg'></img>
<hr />

Trained with (MUCH) help from [<a href="https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7">Andrej Karpathy's Youtube series</a>].

Data was obtained and cleaned from [<a href="https://grail.moe">this website</a>] <3 (thank you for helping me through my schooling days)

All rights to the data go to its respective owners.

<hr />

<b>Model Details</b>
GP-GPT is a 10.8M parameter, unaligned document completer.

<hr />
<b>As of now, here is what I have been able to do:</b>

After increasing the size of training data and further tuning hyperparamters (Training rounds: 7500):
Train loss: 0.98660 | Val loss: 1.27908

GP-GPT: "result or whether "is Massams staunching. Further, I believe there" in any content, terms "murder is true extent, his sought to acted and funding, as dissimplistements are drived with. It with this indeed the construct' has the building to work. Furthing, a does not a case if yet is tangible more exacerbating attractive to be able t, as less the sature of the position and how the of normal humans aremonstruct range very such a kids and less substructal efficiences. "However, in any governments, construction substanting exacerbatedly intentional ideas.
In recence has evidently physics our safety conduction raised title‐sense tasks, many circulated in student, due to murdent the corner and movie society as boggler us from the modern-based - result, sting economic process base, and community of tour mindsets, tools for good stages at the large city of the some of these premises for gender equipment, fewtors and such attempt to be most, capability and hegh‐driven gives of then transminal s"
<br /><br />

After experimenting with the hyperparameters and training it on a tokenizer with a vocab size of 1000 (so 744 merges) and 15000 rounds of training:

Train loss: 2.56862 | Val loss: 3.52850
"l culture oxpenia, "trewoms are" which consequently to what we objective and indiscence have the conflict resorting and want denollictable.
The author"s claim, we are notenance the immense always act in what a precision today, a good governance 'racology's being producuring'. It is element today, where users can others us similar signs and examples which should remain keep condit from the author's place a pressure.
Gerefore, the argument, the charact that works it seems as non-lerent ass all of this could only implies. Singapore should reve them a large amount of the endditional liber of objection, it is ble that Singaporeans has apareness to affect our work of life that hours the governmentandcalance between the behind baby statickes. Mural resomation by the Sponentsward is a bought decadful, it making our face effective to opportunitions, and yet with uniquality of all Singapore. G"

^I suspect this was because (1) the dataset was too small, causing the tokenizer to recognize full words in the dataset and hence perform poorly in validation (which was 10% of the dataset) (2) structure of the model was not optimized to fit a deep network

<hr />

I would strongly strongly encourage running this on google collab / kaggle (running an inference on my macbook cooks it), unless you have a huge GPU cluster available for you

<hr />

<b>Notable to-dos:</b>

1. Increase the size of the dataset
2. Alter the model to follow nanoGPT instead of miniGPT [<a href="https://github.com/karpathy/nanoGPT">By Karpathy too!</a>]
3. Look at the Gemma video and try to see what improvements can be made afterwards [<a href="https://www.youtube.com/watch?v=WW7ZxaC3OtA">From this video</a>]
4. Check out either Grok or DBRX
