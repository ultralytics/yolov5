yarn run v1.22.11
$ jest --coverage
 PASS  tests/index.spec.ts (6.555 s)
  tslint/config
    valid
      ✓ var foo = true; (2579 ms)
      ✓ import { x } from './file'; console.assert(x);  (1476 ms)
      ✓ throw "should be ok because rule is not loaded"; (10 ms)
    invalid
      ✓ throw "err" // no-string-throw (32 ms)
      ✓ var foo = true // semicolon (15 ms)
      ✓ var foo = true // fail (38 ms)
      ✓ const a: string = 1; const sum = 1 + '2';  (16 ms)
  tslint/error
    ✓ should error on missing project (23 ms)
    ✓ should error on default parser (7 ms)
    ✓ should not crash if there are no tslint rules specified (10 ms)


=============================== Coverage summary ===============================
Statements   : 95% ( 38/40 )
Branches     : 95.45% ( 21/22 )
Functions    : 100% ( 10/10 )
Lines        : 95% ( 38/40 )
================================================================================
Test Suites: 1 passed, 1 total
Tests:       10 passed, 10 total
Snapshots:   0 total
Time:        6.825 s
Ran all test suites.
Done in 7.51s.
