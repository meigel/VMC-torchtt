# UQ-ADF (TorchTT)
The Python UQ-ADF reimplementation lives in `vmc_reconstruction/uq_adf_torchtt.py`
and is exercised by the tests/examples in the repo root.

- Run the Darcy example (with convergence plots):
  - `./venv/bin/python examples/uq_adf_darcy_2d.py`
  - Fast mode: `./venv/bin/python examples/uq_adf_darcy_2d.py --fast`
- Run tests:
  - `./venv/bin/python -m pytest -q tests`

#NOTES
- cookie4/ord7_ortho/info.json ist nicht ord7, sondern ord13
- a bug of openblas results in deadlocks in parallel execution using multiprocessing

#TODO
- rewrite `HOW TO`
- remove get_paths

#MODIFICATIONS IN FUTURE VERSIONS
- store moments as m1=..., m2=... in npz as well!
- merge `SampleDump` and `SampleChunks` -- schreibe eine einfache klasse, dass das h5py.File Objekt im NPZ-Fall imitiert,
  sowie einen einfachen Dump-Wrapper, der beide klassen akzeptiert
  (parallel HDF5?)
- extend the new `setup(info)`/`setup_space(info)`  structure to hierarchical Experiments:
    results
    └── cookie3 # experiment name
        ├── info.json
        ├── mc # subexperimant name
        │   ├── info.json
        │   ├── moments.npz
        │   ├── samples.h5
        │   └── reco_50k # subexperiment name
        │       ├── info.json
        │       └── reconstruction.npz
        └── qmc
  dh man muss `stiffness.npz` nicht mehr hin und her linken
  dabei werden dann info-files in lvls geladen. cookie3/info.json und cookie3/mc/info.json sind beide notwendig für `setup`.
  (wobei ersteres genügt für `setup_space`)
  zwei möglichkeiten:
  - setup_space(info1)
    setup_problem(info2)
  - info3 = dict(info1)
    info3.update(info2)
    setup(info3)
  das erste wirkt sauberer...
- Lege eine noSQL-Datenbank an um die Experiment-Ordner zu inidzieren (die Struktur der Ordner ist beliebig und damit verwirrend).
  finde den experimentnamen auf grundlage von tags:
  ```
  $ python lookup.py problem_name=cookie3, sampling_strategy=random, reconstruction_sample_size=50000
  results/cookie3/mc/reco_50k                                                                                _
  ```
  damit muss also jedes feld eindeutig sein (reconstruction sample size statt sample size)
- list_info.py: list all parameters of the specified experiment and all its parents (Gegenstück zu lookup)


#HOW TO (Conversion)

All provided scipts follow the same scheme: <schriptname> PATH SAVE_PATH.
    PATH - base directory for the experiment
    SAVE_PATH - base directory for the output

Consider as an example the follwing directory structure.
Irrelevant files, like logfiles and reconstruction results, are being ignored in the following listing.

    results
    └── cookie3
        ├── samples
        │   ├── 0.npz
        │   ├── 1.npz
        │   ├── 2.npz
        │   ├── 3.npz
        │   ├── 4.npz
        │   ├── 5.npz
        │   └── 6.npz
        ├── info.json
        └── moments.npz

From this tree we want to extract the samples, the moments and the `info.json` file in "Xerus-Format" into the directory `E6`.
To do so just execute the following three commands.

$ python convert/info_to_xerus_format.py results/cookie3 E6.
$ python convert/samples_to_xerus_format.py results/cookie3 E6.
$ python convert/moments_to_xerus_format.py results/cookie3 E6.

Your working directory should now list

    results
    └── cookie3
        ├── samples
        │   ├── 0.npz
        │   ├── 1.npz
        │   ├── 2.npz
        │   ├── 3.npz
        │   ├── 4.npz
        │   ├── 5.npz
        │   └── 6.npz
        ├── info.json
        └── moments.npz

    E6
    ├── 100000m1.dat
    ├── 100000m2.dat
    ├── 10000m1.dat
    ├── 10000m2.dat
    ┊
    ├── 98000m1.dat
    ├── 98000m2.dat
    ├── E6-1000.dat
    ├── E6-1001.dat
    ┊
    ├── E6-99.dat
    ├── E6-9.dat
    └── info.json

If one of these commands does not work as expected it be because the data is provided in the old format.
This is easy to fix but hopefully not necessary. Ask me for a script if needed.


#HOW TO (Reconstruction)

Start a reconstruction with 10000 samples.

For this we need to execute an mc run with arguments `--bs=BS` and `--nl=NL` such that `BS*2**NL >= 10000`
and supply the optional `--dump=10000` flag to tell `run_mc.py` to dump the first 10000 samples.
    $ python run_mc.py [arguments] --dump=10000

Use the `--help` option to get a list of viable arguments.

The convergence rates of the Monte-Carlo estimates can be visualized using
    $ python plot_rates.py [arguments]
the expectation and variance on different levels of the hierarchical MC sampling can be plottet using
    $ python plot_solution.py [arguments]

Now we want to orthogonalize the samples using
    $ python orthogonalize_samples.py [arguments]
For this we need the Cholesky-Factors of the Stiffness matrix.
You can obtain those by executing `dump_stiffness.py` with the same arguments as `run_mc.py`.
    $ python dump_stiffness.py [arguments]

Finally, we can reconstruct the TT-Tensor using
    $ python reco.py [arguments] --ortho
where the `--ortho` flag is optional and indicates that we want to used the orthogonalized samples for reconstruction.

The difference of the reconstructed result to the MC estimates can be visualized with the command
    $ python plot_reco_diff.py [arguments]


#LEGACY FILES

$ cd reconstruction-code; find -name '*_S*' | wc -l
12
$ cd reco_local; find -name 'reconstruction.npz'
9
$ # es fehlen:
$ find -name '*_S*' | egrep 'M10|ord20'
./darcy_problem_M10_N50_deg1_qmc_S10000_ord7_ortho_adf.npz
./darcy_problem_M10_N50_deg1_qmc_S10000_ord7_ortho.npz
./darcy_problem_M20_N50_deg1_qmc_S10000_ord20_ortho.npz

$ cd reconstruction-code; find -name '*-expectation*' | wc -l
10
$ cd reco_local; find -iname 'moments.npz' | wc -l
14
$ # es sind zu viel:
$ find -iname 'moments.npz' | grep test
./results/nonlinear/test/moments.npz
./results/cookie3_test/moments.npz
./results/convection/test/moments.npz
./results/cookie2/test/moments.npz
./results/cookie1/test/moments.npz
$ # es fehlen:
$ find -name '*-expectation*' | grep M10
./darcy_problem_M10_N50_deg1_L3_B2000_qmc-expectation.npz
