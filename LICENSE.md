# License

mind-nerve is distributed under a split license:

## Apache License 2.0 — applies to the repository, the Python wheel, and public-clean weights

The architecture (`spec/`), pure-MIND implementation (`src/`, `cli/`),
integration shims (`integrations/`), Python wheel (`python/mind_nerve/`),
and tests (`tests/`) are licensed under the Apache License, Version 2.0.

The Phase-1 v1.1-oss trained weights published at
<https://huggingface.co/star-ga/mind-nerve-phase1> are also distributed
under Apache 2.0. The training corpus used for those weights is public-clean
(no STARGA-private content) — the model card and `manifest.json` carry the
corpus hash for reproducibility.

You may use, modify, and distribute this code and these weights, including
in commercial products, subject to the terms of the Apache 2.0 license. A
copy is available at <https://www.apache.org/licenses/LICENSE-2.0>.

## STARGA Commercial Terms — applies to `libmindnerve.so` and STARGA-private weights

The Python wheel `mind-nerve` bundles `libmindnerve.so`, a FORTRESS-protected
runtime component. The protected binary is the Phase-2 native inference layer.
Its source is not distributed and is licensed under STARGA Commercial terms;
the Phase-1 PyTorch path published here does not depend on it.

Future weights produced from STARGA-private training corpora (for example,
weights tuned on internal customer skill catalogs) are not Apache-2.0 and are
distributed, if at all, under separate STARGA Commercial agreements. Such
weights are clearly labelled at release time; the Phase-1 v1.1-oss weights
described above are Apache.

Independently trained weights produced by third parties using this Apache-2.0
architecture are not constrained by these terms.

For commercial deployments needing per-customer FORTRESS-locked builds of the
runtime layer, contact `license@star.ga`.

## Patent grant

The Apache 2.0 license includes an express patent grant covering contributions
made to this repository and to the Apache-licensed Phase-1 weights. That grant
does not extend to the protected `libmindnerve.so` binary or to STARGA-private
weights produced from internal training corpora.

## Trademark

"STARGA", "mind-nerve", "mind-runtime", and "MIND" are trademarks of STARGA Inc.
Use of these marks is governed by STARGA's trademark policy, not by the
Apache 2.0 license.

## Contact

STARGA Inc. — info@star.ga · license inquiries: license@star.ga
