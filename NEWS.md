2.0.4
=====
- add  preset selection functions for Schur decomposition. This avoids the cfunction closure that doesn´t work on all plateforms

2.0.3
=====
- removes cfunction closure

2.0.2
=====
- permits zero matrix in all QR functions

2.0.1
=====
- fix bug in resize!() for real eigenvalues
- increase size of test matrices

2.0.0
=====
- add MKL compatibility for ormqr!()
- add QROrmWs now required for ormqr! [breaking change]

1.2.9
=====
- fix link to libblastrampoline for Windows in Julia-1.9

1.2.8
=====
- fix generalized Schur ws.eigen_values

1.2.7
=====
- lse implementation

1.2.6
=====
- implement getrs!
- fix complex numbers QRPivotedWs

1.2.5
=====
- fix resizing
- don't return views

1.2.4
=====
- fix resizing

1.2.3
=====
- adding support for orgqr! and orgql!

1.2.2
=====
- fix method redefinition

1.2.1
=====
- fix method defintions

1.2.0
=====
- add resizing of workspace

1.1.0
=====
- fix eigenvalues

1.0.0
=====
- total refactoring

0.1.3
=====
- fix pivoted QR

0.1.1
=====
- adding pivoted QR
- several fixes

0.1.0
=====
- initial files
