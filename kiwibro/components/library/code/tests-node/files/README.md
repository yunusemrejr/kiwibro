# {{name}}

A starting point for tests in Node, using `node:test` and `node:assert`: no
installing, no config file.

    src/text.js          the sample module under test
    test/text.test.js    table-driven cases, one async case and one rejection

## Test

    node --test

Each row of a table is one test, named after its case, so a failing row says which
input broke. Replace `src/text.js` and its cases with the project's own code, keep
the shape: a table for the cases that only differ in data, a plain test for the
ones that differ in behaviour.

`node --test --watch` runs the tests again on every save.
