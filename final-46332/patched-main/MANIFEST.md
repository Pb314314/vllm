# Exact files the "after" arms ran (copied into the worktree by set_arm)

These are byte-identical to the PR head `75d5447a1`'s blobs:

| file | git blob id (at 75d5447a1) | sha256 |
|---|---|---|
| shm_broadcast.py | f298580736010a52b947f01d946e01e172c41177 | c5454af1d3fc4d457e08e6e7ee84a1f8f02ef90ae121ddfb3b3c8d1cd7be30e5 |
| multiproc_executor.py | 21881f3473616e782755119d368213afc06bcc4c | 06c38a19fa19ecf5ecfc0eec006ecf85ccd516f1e6eaa64bb2dc0cf17896044b |
| ray_executor_v2.py | 425cf2a28ab5cc8d5ccf95facc45b40d09b04c8e | 8d493f9ebbdc5d3f58f15b386f89b11ee3afe36fba14f49061d6439a7374acd8 |

Verify: `git cat-file blob <blob-id> | sha256sum` against the sha256 column.

# Resolved binary wheel (recovered by hash match)

The precompiled install resolved to the published nightly wheel
`vllm-0.33.0rc1.dev946+gb2094acc8-cp38-abi3-manylinux_2_28_aarch64.whl`
(commit `b2094acc88a7f76b49332ebc91a34887272e36c8`, default variant):
https://wheels.vllm.ai/b2094acc88a7f76b49332ebc91a34887272e36c8/vllm-0.33.0rc1.dev946%2Bgb2094acc8-cp38-abi3-manylinux_2_28_aarch64.whl

`wheel/wheel-ext-manifest.txt` has sha256 of every `.abi3.so` in that wheel;
`final-46332/purge-after/loaded_ext.sha256` (hashed from the live serve's
/proc/*/maps) matches it for all 8 mapped extensions — including the three
nested ones absent from the install-time top-level glob
(`third_party/deep_gemm/_C_extension.abi3.so`, `vllm_flash_attn/_vllm_fa2_C`,
`vllm_flash_attn/_vllm_fa3_C`). Zero conflicts.
