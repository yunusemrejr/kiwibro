# {{name}}

A small HTTP JSON API in Node.js. No dependencies: `node:http` serves the routes,
`node:test` runs the tests.

## Routes

    GET  /health        {"status":"ok","notes":1}
    GET  /notes         every note
    POST /notes         {"title":"...","body":"..."} -> 201
    GET  /notes/:id     one note

Every reply is JSON. A bad request answers 400, an unknown note or route 404 and a
wrong method 405, each with an `error` field that names what was expected.

## Run

    node src/server.js

`PORT` changes the port (default 8080).

## Test

    node --test

The tests start the server on a free port, so nothing else may be listening.
