import tensorflow as tf
import tensorflow_hub as hub


def TestWeightedLayers(tokens_input, tokens_length):
    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    embeddings = elmo(inputs={"tokens": tokens_input,"sequence_len": tokens_length},
                  signature="tokens",
                  as_dict=True)["elmo"]
    print(embeddings)
    return embeddings
    