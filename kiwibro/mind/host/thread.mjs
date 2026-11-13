// Helper thread of the Kiwi Mind engine: shares the main thread's WebAssembly.Memory and sleeps inside the
// engine (memory.atomic.wait32) until matrix rows or attention heads are handed out. Derived from the
// Flyingkiwi helper thread by the same author.
import { parentPort, workerData } from 'node:worker_threads';

const { module, memory, index, stackTop } = workerData;
const instance = new WebAssembly.Instance(module, { env: { memory } });
instance.exports.__stack_pointer.value = stackTop;
parentPort.postMessage('ready');
instance.exports.km_thread_main(index); // returns only when the worker is terminated
