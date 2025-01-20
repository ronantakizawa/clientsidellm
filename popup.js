import { pipeline, env } from './dist/transformers.min.js';

// Performance tracking class
class PerformanceTracker {
    constructor() {
        this.marks = new Map();
        this.measures = new Map();
        this.startTime = performance.now();
    }

    mark(name) {
        this.marks.set(name, performance.now());
        console.log(`🕒 ${name} started at ${Math.round(performance.now() - this.startTime)}ms`);
    }

    measure(name, startMark, endMark) {
        const start = this.marks.get(startMark);
        const end = this.marks.get(endMark);
        if (start && end) {
            const duration = end - start;
            this.measures.set(name, duration);
            console.log(`⏱️ ${name}: ${Math.round(duration)}ms`);
        }
    }

    getSummary() {
        const measures = Array.from(this.measures.entries())
            .sort((a, b) => b[1] - a[1])
            .map(([name, duration]) => `${name}: ${Math.round(duration)}ms`)
            .join('\n');
        return measures;
    }
}

// Configure environment with optimizations
const extensionUrl = chrome.runtime.getURL('');
env.localModelPath = extensionUrl + 'models/';
env.remoteModels = false;
env.allowLocalModels = true;
env.backends.onnx.wasm.wasmPaths = extensionUrl + 'dist/';

// Optimize ONNX runtime configuration
env.backends.onnx.wasmConfig = {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
    enableMemPattern: true,
    executionMode: 'sequential',
    numThreads: navigator.hardwareConcurrency || 4,
    simd: true,
    logLevel: 'error'
};

// Configure worker
env.backends.onnx.wasmWorkerConfig = {
    type: 'module',
    credentials: 'same-origin'
};

const textToSummarize = `
The field of artificial intelligence has seen remarkable progress in recent years. 
Deep learning models have achieved human-level performance in tasks like image recognition, 
natural language processing, and game playing. The development of transformer architectures 
in particular has revolutionized how AI processes sequential data. These advances have led 
to breakthroughs in machine translation, text generation, and conversational AI. However, 
challenges remain in areas like common sense reasoning and generalizing from limited data.
`;

let summarizer = null;
let isProcessing = false;
let isInitialized = false;
let performanceTracker = new PerformanceTracker();

async function cleanupResources() {
    if (summarizer) {
        try {
            performanceTracker.mark('cleanup_start');
            await summarizer.dispose();
            summarizer = null;
            isInitialized = false;
            performanceTracker.mark('cleanup_end');
            performanceTracker.measure('Cleanup Time', 'cleanup_start', 'cleanup_end');
        } catch (error) {
            console.error('Error during cleanup:', error);
        }
    }
}

async function runSummarization() {
    if (isProcessing || isInitialized) return;
    
    const statusElement = document.getElementById('status');
    const summaryElement = document.getElementById('summary');
    
    isProcessing = true;
    performanceTracker = new PerformanceTracker();
    
    try {
        statusElement.textContent = 'Generating output...';
        performanceTracker.mark('init_start');

        performanceTracker.mark('pipeline_start');
        summarizer = await pipeline(
            'summarization',
            'bart-large-cnn',
            {
                quantized: true,
                skipMissing: true,
                progress_callback: (progress) => {
                    // Only log to console
                    if (progress.status === 'done') {
                        console.debug('Loaded:', progress.file);
                    }
                }
            }
        );
        performanceTracker.mark('pipeline_end');
        performanceTracker.measure('Pipeline Initialization', 'pipeline_start', 'pipeline_end');
        
        isInitialized = true;
        console.log('Model loaded, generating summary...');
        statusElement.textContent = 'Generating output...';
        
        performanceTracker.mark('inference_start');
        const result = await summarizer(textToSummarize, {
            max_length: 100,
            min_length: 30,
            do_sample: false,
            num_beams: 1,
            early_stopping: true,
            temperature: 0.3,
            top_p: 0.9,
            no_repeat_ngram_size: 3
        });
        performanceTracker.mark('inference_end');
        performanceTracker.measure('Inference Time', 'inference_start', 'inference_end');
        
        statusElement.textContent = 'Summary generated!';
        summaryElement.textContent = result[0].summary_text;

        console.log('\n📊 Performance Summary:\n' + performanceTracker.getSummary());
        
    } catch (error) {
        console.error('Pipeline error:', error);
        statusElement.textContent = 'Error generating summary. Please try again.';
        isInitialized = false;
    } finally {
        performanceTracker.mark('total_end');
        performanceTracker.measure('Total Time', 'init_start', 'total_end');
        isProcessing = false;
    }
}

// Error handlers
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
});

window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled rejection:', event.reason);
});

// Cleanup handlers
window.addEventListener('unload', cleanupResources);
chrome.runtime.onSuspend?.addListener(cleanupResources);

// Initialize once
console.log('Starting summarization process...');
runSummarization();