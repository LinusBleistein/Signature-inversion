# Signature-inversion

This code implements the invertion algorithm of Chang and Lyons (2019) to invert signatures.

Given the signature `sig` of a path in R^d, truncated at depth n, you can run the following command to reconstruct the path

```
from insertion import invert_signature

inverted_path=invert_signature(sig,n,d)
```
Note that `sig` should not contain the constant term 1, as is done by default in the signatory package: `sig=signatory.signature(path,n)`. Note also that the inverted path is obtained up to translations. Information about the initial position of the path x0 may be added as an extra argument:

```
inverted_path=invert_signature(sig,n,d,first_point=x0)
```