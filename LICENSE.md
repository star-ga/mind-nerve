# License

mind-nerve is distributed under a split license:

## Apache License 2.0 — applies to the repository, the Python wheel, and public-clean weights

The architecture (`spec/`), pure-MIND implementation (`src/`, `cli/`),
integration shims (`integrations/`), Python wheel (`python/mind_nerve/`),
and tests (`tests/`) are licensed under the Apache License, Version 2.0.

The Phase-1 v1.1-oss trained weights published at
<https://huggingface.co/star-ga/mind-nerve> are also distributed
under Apache 2.0. The training corpus used for those weights is public-clean
(no STARGA-private content) — the model card and `manifest.json` carry the
corpus hash for reproducibility.

You may use, modify, and distribute this code and these weights, including
in commercial products, subject to the terms of the Apache 2.0 license. A
copy is available at <https://www.apache.org/licenses/LICENSE-2.0>.

## Bundled binary component — separately licensed

The Python wheel `mind-nerve` bundles `libmindnerve.so`, a compiled runtime
component whose source is not part of this repository and is not distributed
under Apache 2.0. The bundled binary remains the property of STARGA Inc.
and is licensed for use as part of the Apache-2.0 wheel surface only;
redistribution of the binary outside that wheel is not granted by this
LICENSE. The Phase-1 PyTorch inference path published here works without
that binary.

Weights produced from non-public training corpora may be released under
separate STARGA agreements rather than Apache 2.0; such weights are clearly
labelled at release time. The Phase-1 `v1.1-oss` weights referenced above
are Apache.

Independently trained weights produced by third parties using this
Apache-2.0 architecture are not constrained by these terms.

For commercial enquiries, contact `license@star.ga`.

## Patent grant

The Apache 2.0 license includes an express patent grant covering contributions
made to this repository and to the Apache-licensed Phase-1 weights. That grant
does not extend to the bundled `libmindnerve.so` binary or to weights
released under separate STARGA agreements.

## Trademark

"STARGA", "mind-nerve", "mind-runtime", and "MIND" are trademarks of STARGA Inc.
Use of these marks is governed by STARGA's trademark policy, not by the
Apache 2.0 license.

## Contact

STARGA Inc. — info@star.ga · license inquiries: license@star.ga
