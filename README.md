# Signature-inversion

This code implements the insertion algorithm of Chang and Lyons (2019) to invert signatures.

Given the signature `sig` of a batch of paths, truncated at a certain depth, run the following command to reconstruct the path:

```
from insertion import invert_signature

inverted_path = invert_signature(sig, depth, d)
```
Note that `sig` should not contain the constant term 1, as is done by default in the signatory package. In other words, `sig`should be the output of `signatory.signature(path, depth)`. Note also that the inverted path is obtained up to translations. Information about the initial position of the path x0 may be added as an extra argument with:

```
inverted_path = invert_signature(sig, depth, d, initial_position=x0)
```

Some examples are given in the notebook `Example.ipynb`.


## Environment

The packages are listed in `requirements.txt`. Run the following two lines to set up the environment (signatory must be installed after PyTorch):

```
pip install -r requirements.txt
pip install signatory==1.2.2.1.5.0 --no-cache-dir --force-reinstall
```


## References


[1]: Chang, J. and Lyons, T. (2019) [Insertion algorithm for inverting the signature of a path](https://arxiv.org/abs/1907.08423)