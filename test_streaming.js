const puppeteer = require('puppeteer');

async function testStreamingUI() {
    const browser = await puppeteer.launch({ 
        headless: true, 
        slowMo: 100,
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    
    const page = await browser.newPage();
    
    // Set viewport size
    await page.setViewport({ width: 1200, height: 800 });
    
    try {
        console.log('🚀 Starting Streamlit UI test...');
        
        // Navigate to the Streamlit app
        await page.goto('http://localhost:8501', { 
            waitUntil: 'networkidle2',
            timeout: 30000 
        });
        
        console.log('✅ Page loaded successfully');
        
        // Take initial screenshot
        await page.screenshot({ path: 'initial_state.png' });
        console.log('📸 Initial screenshot taken');
        
        // Wait for the app to fully load
        await page.waitForSelector('[data-testid="stApp"]', { timeout: 10000 });
        console.log('✅ Streamlit app detected');
        
        // Look for the model selection
        await page.waitForSelector('[data-testid="stMultiSelect"]', { timeout: 5000 });
        console.log('✅ Model selector found');
        
        // Check if GraphRAG v2.1 is selected by default
        const selectedModels = await page.$$eval('[data-testid="stMultiSelect"] [data-baseweb="tag"]', 
            elements => elements.map(el => el.textContent)
        );
        console.log('📊 Selected models:', selectedModels);
        
        // Find the chat input - try multiple selectors
        let chatInput;
        try {
            chatInput = await page.waitForSelector('[data-testid="stChatInput"] textarea', { timeout: 3000 });
            console.log('✅ Chat input found (method 1)');
        } catch (e) {
            try {
                chatInput = await page.waitForSelector('textarea[placeholder*="Ask me anything"]', { timeout: 3000 });
                console.log('✅ Chat input found (method 2)');
            } catch (e2) {
                try {
                    chatInput = await page.waitForSelector('textarea', { timeout: 3000 });
                    console.log('✅ Chat input found (method 3)');
                } catch (e3) {
                    console.log('❌ Chat input not found with any method');
                    throw e3;
                }
            }
        }
        
        // Type a test question
        const testQuestion = 'what is this?';
        await chatInput.type(testQuestion);
        console.log(`💬 Typed question: "${testQuestion}"`);
        
        // Take screenshot before submitting
        await page.screenshot({ path: 'before_submit.png' });
        console.log('📸 Before submit screenshot taken');
        
        // Submit the question
        await chatInput.press('Enter');
        console.log('⏳ Question submitted, monitoring streaming...');
        
        // Monitor for streaming elements
        const streamingTests = [];
        
        // Test 1: Check if "Thinking" header appears
        try {
            await page.waitForSelector('div:has-text("Thinking")', { timeout: 5000 });
            streamingTests.push('✅ "Thinking" header appeared');
        } catch (e) {
            streamingTests.push('❌ "Thinking" header not found');
        }
        
        // Test 2: Monitor streaming content changes
        let previousContent = '';
        let streamingChanges = 0;
        
        for (let i = 0; i < 20; i++) {
            await new Promise(resolve => setTimeout(resolve, 500));
            
            try {
                const currentContent = await page.evaluate(() => {
                    const elements = document.querySelectorAll('[data-testid="stMarkdownContainer"]');
                    return Array.from(elements).map(el => el.textContent).join('\n');
                });
                
                if (currentContent !== previousContent && currentContent.includes('Initializing')) {
                    streamingChanges++;
                    console.log(`🔄 Streaming change ${streamingChanges} detected at ${i * 0.5}s`);
                    
                    // Take screenshot of streaming state
                    await page.screenshot({ path: `streaming_${streamingChanges}.png` });
                }
                
                previousContent = currentContent;
                
                // Check if streaming is complete
                if (currentContent.includes('Response synthesis complete') || 
                    currentContent.includes('Analysis complete')) {
                    console.log('✅ Streaming completed');
                    break;
                }
            } catch (e) {
                console.log(`⚠️ Error checking content at ${i * 0.5}s:`, e.message);
            }
        }
        
        streamingTests.push(`🔄 Detected ${streamingChanges} streaming changes`);
        
        // Test 3: Check final state
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        try {
            await page.waitForSelector('[data-testid="stSuccess"]', { timeout: 5000 });
            streamingTests.push('✅ Success message appeared');
        } catch (e) {
            streamingTests.push('❌ Success message not found');
        }
        
        // Take final screenshot
        await page.screenshot({ path: 'final_state.png' });
        console.log('📸 Final screenshot taken');
        
        // Test 4: Check if answer appears
        const finalContent = await page.evaluate(() => {
            const elements = document.querySelectorAll('[data-testid="stMarkdownContainer"]');
            return Array.from(elements).map(el => el.textContent).join('\n');
        });
        
        if (finalContent.includes('Knowledge Counselor')) {
            streamingTests.push('✅ Answer appeared');
        } else {
            streamingTests.push('❌ Answer not found');
        }
        
        // Print test results
        console.log('\n🧪 STREAMING TEST RESULTS:');
        console.log('================================');
        streamingTests.forEach(test => console.log(test));
        
        console.log('\n📊 FINAL CONTENT PREVIEW:');
        console.log(finalContent.substring(0, 500) + '...');
        
        // Check spacing issues
        const spacingIssues = await page.evaluate(() => {
            const elements = document.querySelectorAll('[data-testid="stMarkdownContainer"]');
            let issues = [];
            
            elements.forEach((el, index) => {
                const style = window.getComputedStyle(el);
                const marginBottom = parseFloat(style.marginBottom);
                const marginTop = parseFloat(style.marginTop);
                
                if (marginBottom > 20 || marginTop > 20) {
                    issues.push(`Element ${index}: margin-bottom ${marginBottom}px, margin-top ${marginTop}px`);
                }
            });
            
            return issues;
        });
        
        if (spacingIssues.length > 0) {
            console.log('\n⚠️  SPACING ISSUES DETECTED:');
            spacingIssues.forEach(issue => console.log(issue));
        } else {
            console.log('\n✅ No major spacing issues detected');
        }
        
    } catch (error) {
        console.error('❌ Test failed:', error);
        await page.screenshot({ path: 'error_state.png' });
    } finally {
        await browser.close();
        console.log('🔚 Browser closed');
    }
}

// Run the test
testStreamingUI().catch(console.error);