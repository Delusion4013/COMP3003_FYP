## Environment preparation




## Need re-generate

- Whole
  - [ ] 4pce
  - [x] 4uiz
  - [x] 5h21
  - [x] 5z5v
  - [x] 6q3y
  
- Partial
  - 2yel
    - [x] fld02-407
    - [x] fld02-629
    - [x] fld01-630
  - 3u5l
    - [x] fld02-396
    - [x] fld01-616
  - 3zyu
    - [x] fld02-393
    - [x] fld02-612
  - 4bw2
    - [x] fld01- 96~99
  - 4cfl
    - [x] fld01-950
    - [x] fld02-998
    - [x] fld02-950
    - [x] fld02-975
    - [x] fld01-998
    - [x] fld01-977
  - 4clb
    - [x] 02-954
    - [x] 01-955
    - [x] 01-980
    - [x] 02-981
  - 4gpj
    - [x] 02-940
    - [x] 02-988
    - [x] 02-965
    - [x] 01-989
    - [x] 02-967
    - [x] 01-941
    - [x] 01-966
  - 4hbw
    - [x] 01-449
    - [x] 02-448
    - [x] 01-497
    - [x] 01-474
    - [x] 02-496
    - [x] 02-497
    - [x] 02-475
  - 4hxl
    - [x] 01-879
    - [x] 02-878
    - [x] 01-927
    - [x] 02-904
    - [x] 01-906
    - [x] 02-927
  - 4o7e
    - [x] 01-53
  - 6q3y
    - [x] 01-356
  - from 6cj1

## Reference

- Commands

- \1. Open the Open3DQSAR software program.

   

  \2. Load one of the SDF files from the data set with the reduced sampling rate (trajectories_aligned_reduced_max_min):

  Eg. import type=SDF file=C:\trajectories_aligned_reduced_max_min\2yel.sdf

   

  \3. Setup a MIF grid size. I've determined that a 0.333 Angstrom step size is probably the optimal balance for our needs here:

  Ie. box step=0.333

   

  \4. Calculate the steric MIF:

  Ie. calc_field type=VDW

   

  \5. Calculate the electrostatic MIF:

  Ie. calc_field type=MM_ELE

   

  \6. Export the MIF data as ASCII files:

  Ie. export type=object_field format=xyz file=C:\ object_list=all

   

  \7. Find the MIF files and delete the last six of them for both the VDW and ELE MIFs, as these are just there to make the grids the same size, and are not part of the data set:

  Ie. Find and delete the MIFs with names "object_list=all_fld-01_obj-1002" to "object_list=all_fld-01_obj-1007" and "object_list=all_fld-02_obj-1002" to "object_list=all_fld-02_obj-1007".

   

  \8. Repeat steps 1-7 for all 72 SDF files.

- [Downloading RDKit](https://www.rdkit.org/docs/Install.html)