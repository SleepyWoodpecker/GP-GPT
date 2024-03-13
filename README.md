Trained with (MUCH) help from [<a href="https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=7">Andrej Karpathy's Youtube series</a>].

Data was obtained and cleaned from [<a href="https://grail.moe">this website</a>] <3 (thank you for helping me through my schooling days)

All rights to the data go to its respective owners.

<b>Model Details</b>
GP-GPT is a 10.8M parameter, unaligned document completer.

<b>As of now, here is what I have been able to do:</b>

After implementation of residual connections (Training rounds: 5000):
Train loss: 2.03122 | Val loss: 2.09755

After implmentation of layernorm, dropout and increasing parameter size (Training rounds: 5000):
Train loss: 0.89576 | Val loss: 1.28290

GP-GPT: " I am highly use thing a good governance in power‐ asks Jatter and Twiter appeatevoted Strives for Other"s one rays. When his assum to our tectional crasts that it makes how a bas/hour acquisitive and internations of certain is defining. This is threats the nature and streate to different get to be a formed. US that found do other is a strenning that tree world today isn't invalided someone that it. If the governments' can hope to do live, but they predict for us their giveable. For indictors to recount current significance technology and so keep the unique toway given this virtue. Show that may become reaching that justifying a bogramous experience difficuls (much still easier to fave and refutal or suppered victimation enably to acques and their gried for corage. Asing it to be fundamental effective of disorstates, but also in either premises. Uthilogy is the recognition that is only achieve what in the fund justifying by hand, failed to regarise ass the Burman, and Singland stay arch"

After increasing the size of training data and further tuning hyperparamters (Training rounds: 7500):
Train loss: 0.98660 | Val loss: 1.27908

GP-GPT: "result or whether "is Massams staunching. Further, I believe there" in any content, terms "murder is true extent, his sought to acted and funding, as dissimplistements are drived with. It with this indeed the construct' has the building to work. Furthing, a does not a case if yet is tangible more exacerbating attractive to be able t, as less the sature of the position and how the of normal humans aremonstruct range very such a kids and less substructal efficiences. "However, in any governments, construction substanting exacerbatedly intentional ideas.
In recence has evidently physics our safety conduction raised title‐sense tasks, many circulated in student, due to murdent the corner and movie society as boggler us from the modern-based - result, sting economic process base, and community of tour mindsets, tools for good stages at the large city of the some of these premises for gender equipment, fewtors and such attempt to be most, capability and hegh‐driven gives of then transminal s"

No wiser words have been said tbh.

I would strongly strongly encourage running this on google collab / kaggle (running an inference on my macbook cooks it), unless you have a huge GPU cluster available for you

<b>Notable to-dos:</b>

1. Increase the size of the dataset
2. Improve the tokenizer (perhaps the tokenizer video from the same series)
3. Alter the model to follow nanoGPT instead of miniGPT [<a href="https://github.com/karpathy/nanoGPT">By Karpathy too!</a>]
4. Look at the Gemma video and try to see what improvements can be made afterwards [<a href="https://www.youtube.com/watch?v=WW7ZxaC3OtA">From this video</a>]
