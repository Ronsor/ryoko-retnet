# Ryoko

Ryoko ("Retention yields optimized knowledge output") is an implementation of a causal language model
based on a modified implementation of Retentive Networks (https://arxiv.org/pdf/2307.08621.pdf). The
implementation in `ryoko/retention.py` is a 100% self-contained, correct implementation of multi-scale
retention as described in the paper. The API provided is largely based on PyTorch's `nn.MultiheadAttention`.
Implementation modifications will be placed in separate source files and clearly marked.

In the future, pretrained weights will be available for testing, along with proper scripts for training
and inference.

# License

Copyright &copy; 2023 Ronsor Labs. All rights reserved.

This code and accompanying pretrained model weights, if any, are provided to you under the terms of
the Free Development Public License 1.0-US, the text of which may be found in the `LICENSE` file at the
root of this repository or at <https://freedevproject.org/fdpl-1.0-us>.
