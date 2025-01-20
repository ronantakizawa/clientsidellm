import { pipeline, env } from './dist/transformers.min.js';

// Configure for local model
const extensionUrl = chrome.runtime.getURL('');
env.localModelPath = extensionUrl + 'models/';
env.remoteModels = false;
env.allowLocalModels = true;
env.backends.onnx.wasm.wasmPaths = extensionUrl + 'dist/';

const textToSummarize = `
The field of artificial intelligence has seen remarkable progress in recent years. 
Deep learning models have achieved human-level performance in tasks like image recognition, 
natural language processing, and game playing. The development of transformer architectures 
in particular has revolutionized how AI processes sequential data. These advances have led 
to breakthroughs in machine translation, text generation, and conversational AI. However, 
challenges remain in areas like common sense reasoning and generalizing from limited data.
`;

async function runSummarization() {
   const statusElement = document.getElementById('status');
   const summaryElement = document.getElementById('summary');
   
   try {
       statusElement.textContent = 'Loading summarization model...';
       console.log('Starting pipeline with local model');
       
       // Track overall progress
       let filesToLoad = 0;
       let filesLoaded = 0;
       let currentProgress = {};

       const summarizer = await pipeline(
           'summarization',
           'bart-large-cnn',  // Local model name
           {
               quantized: true,
               progress_callback: (progress) => {
                   // Initialize file count on first load
                   if (progress.status === 'download' && !currentProgress[progress.file]) {
                       filesToLoad++;
                       currentProgress[progress.file] = 0;
                   }
                   
                   // Update progress for current file
                   if (progress.status === 'progress' && progress.progress) {
                       currentProgress[progress.file] = progress.progress;
                   }
                   
                   // Mark file as complete
                   if (progress.status === 'done') {
                       currentProgress[progress.file] = 100;
                       filesLoaded++;
                   }
                   
                   // Calculate overall progress
                   const totalProgress = Object.values(currentProgress).reduce((a, b) => a + b, 0);
                   const overallProgress = Math.round((totalProgress / (filesToLoad * 100)) * 100);
                   
                   // Update status
                   const message = `Loading model: ${overallProgress}% complete`;
                   console.log(message);
                   statusElement.textContent = message;
               }
           }
       );
       
       console.log('Model loaded, generating summary...');
       statusElement.textContent = 'Generating summary...';
       
       const result = await summarizer(textToSummarize, {
           max_length: 150,  // Longer summary
           min_length: 40,   // Minimum length
           do_sample: true,  // Enable sampling
           temperature: 0.7  // Add some creativity
       });
       
       console.log('Summary generated:', result);
       statusElement.textContent = 'Summary generated!';
       summaryElement.textContent = result[0].summary_text;
       
   } catch (error) {
       console.error('Pipeline error:', {
           message: error.message,
           stack: error.stack,
           name: error.name
       });
       statusElement.textContent = `Error: ${error.message}`;
   }
}

// Add global error handlers
window.addEventListener('error', (event) => {
   console.error('Global error:', event.error);
});

window.addEventListener('unhandledrejection', (event) => {
   console.error('Unhandled rejection:', event.reason);
});

console.log('Starting summarization process...');
runSummarization();