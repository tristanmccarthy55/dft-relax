# dft-relax

Read `README.md` first: the design (one shared site block + fixed cell), the numbers the starting
structures already give, and the caveats.

Standing rules:
- Everything for this experiment lives here. Do not add to `../ptychoshelves-clean` or `~/atomfind`;
  the builder only reads their outputs.
- CASTEP runs on Blythe via `submit_blythe.sh`, and the user runs it. Give exact commands, each with
  its `scp -O` pull line. Local work uses `~/hyperspy-bundle/bin/python`.
- The reference is the RELAXED GT. Score per axis and on the Ti off-centring, never one all-atom RMS.
- Headline only what was measured. Label the box (periodic 2x2x12 block, no SrTiO3) whenever a
  relaxed structure is described.
