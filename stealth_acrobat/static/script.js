document.addEventListener('DOMContentLoaded', () => {

    // Export tests button
    // const exportBtn = document.getElementById('export-tests-btn');
    // if (exportBtn) {
    //     exportBtn.addEventListener('click', async () => {
    //         try {
    //             const resp = await fetch('/api/tests/export');
    //             if (!resp.ok) {
    //                 console.error('Export failed:', resp.status);
    //                 return;
    //             }
    //             const blob = await resp.blob();
    //             const url = URL.createObjectURL(blob);
    //             const a = document.createElement('a');
    //             a.href = url;
    //             a.download = 'all_tests.zip';
    //             document.body.appendChild(a);
    //             a.click();
    //             a.remove();
    //             URL.revokeObjectURL(url);
    //         } catch (err) {
    //             console.error('Export error:', err);
    //         }
    //     });
    // }

    // Zoom functionality
    let currentZoom = 1.0;

    function applyZoom() {
        const pages = document.querySelectorAll('.pdf-page');
        pages.forEach(page => {
            page.style.zoom = currentZoom;
        });
    }

    const documentViewer = document.querySelector('.document-viewer');

    function zoomAroundCenter(newZoom) {
        if (!documentViewer) {
            currentZoom = newZoom;
            applyZoom();
            return;
        }

        // Calculate dynamic max zoom based on container width vs page width (850px)
        const widthMaxZoom = (documentViewer.clientWidth - 40) / 850;
        const maxZoom = Math.max(1.0, widthMaxZoom); // Allow it to be at least 1.0 (100%), but cap it by screen width if screen is wider
        newZoom = Math.min(Math.max(newZoom, 0.5), widthMaxZoom);

        const rect = documentViewer.getBoundingClientRect();
        // Zoom relative to the center of the visible area
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;

        const scrollLeft = documentViewer.scrollLeft;
        const scrollTop = documentViewer.scrollTop;

        const unzoomedX = (scrollLeft + centerX) / currentZoom;
        const unzoomedY = (scrollTop + centerY) / currentZoom;

        currentZoom = newZoom;
        applyZoom();

        documentViewer.scrollLeft = (unzoomedX * currentZoom) - centerX;
        documentViewer.scrollTop = (unzoomedY * currentZoom) - centerY;
    }

    const zoomInBtn = document.querySelector('.zoom-in');
    const zoomOutBtn = document.querySelector('.zoom-out');

    if (zoomInBtn) {
        zoomInBtn.addEventListener('click', () => {
            zoomAroundCenter(currentZoom + 0.1);
        });
    }

    if (zoomOutBtn) {
        zoomOutBtn.addEventListener('click', () => {
            zoomAroundCenter(currentZoom - 0.1);
        });
    }

    // Ctrl + Mouse Wheel Zoom
    if (documentViewer) {
        documentViewer.addEventListener('wheel', (e) => {
            if (e.ctrlKey) {
                e.preventDefault(); // Prevent default browser zoom or scroll

                // Calculate mouse position relative to the scrollable container
                const rect = documentViewer.getBoundingClientRect();
                const mouseX = e.clientX - rect.left;
                const mouseY = e.clientY - rect.top;

                // Current scroll offsets relative to the old zoom level
                const scrollLeft = documentViewer.scrollLeft;
                const scrollTop = documentViewer.scrollTop;

                // Mouse position mapped to unzoomed coordinates
                const unzoomedX = (scrollLeft + mouseX) / currentZoom;
                const unzoomedY = (scrollTop + mouseY) / currentZoom;

                if (e.deltaY < 0) {
                    const widthMaxZoom = (documentViewer.clientWidth - 40) / 850;
                    currentZoom = Math.min(currentZoom + 0.1, widthMaxZoom);
                } else {
                    currentZoom = Math.max(currentZoom - 0.1, 0.5);
                }
                applyZoom();

                // Compute new scroll offsets so the unzoomed coordinates stay under the mouse
                documentViewer.scrollLeft = (unzoomedX * currentZoom) - mouseX;
                documentViewer.scrollTop = (unzoomedY * currentZoom) - mouseY;
            }
        }, { passive: false }); // Needs to be passive: false to allow e.preventDefault()
    }

    // Abort controller for canceling ongoing connections when the user re-pastes
    let currentAbortController = null;

    // Intercept paste anywhere on the document
    document.addEventListener('paste', async (e) => {
        let pastedText = (e.clipboardData || window.clipboardData).getData('text');
        if (!pastedText) return;

        // Visual feedback
        document.getElementById('loading').style.display = 'flex';

        // Remove old progress if exists
        let oldProg = document.getElementById('dl-progress');
        if (oldProg) oldProg.remove();

        // Removed premature destruction of dummy UI. It stays until results arrive.

        // Cancel the previous fetch if it's still running
        if (currentAbortController) {
            currentAbortController.abort();
            currentAbortController = null;
            console.log("Aborted previous stream due to new paste.");
        }

        currentAbortController = new AbortController();

        try {
            const response = await fetch('/api/solve_stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: pastedText }),
                signal: currentAbortController.signal
            });

            if (!response.body) {
                throw new Error("ReadableStream not supported");
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");

            let buffer = "";
            let done = false;

            while (!done) {
                const { value, done: readerDone } = await reader.read();
                done = readerDone;
                if (value) {
                    buffer += decoder.decode(value, { stream: true });
                    // Process complete SSE chunks
                    while (buffer.includes("\n\n")) {
                        const splitIdx = buffer.indexOf("\n\n");
                        const chunk = buffer.slice(0, splitIdx);
                        buffer = buffer.slice(splitIdx + 2);

                        if (chunk.startsWith("data: ")) {
                            const dataStr = chunk.slice(6);
                            try {
                                const parsed = JSON.parse(dataStr);
                                handleStreamEvent(parsed);
                            } catch (err) {
                                console.error("Error parsing stream JSON", err);
                            }
                        }
                    }
                }
            }
        } catch (err) {
            if (err.name === 'AbortError') {
                console.log('Stream fetch aborted explicitly.');
            } else {
                console.error('Network Error:', err);
                document.getElementById('loading').style.display = 'none';
            }
        } finally {
            if (currentAbortController && !currentAbortController.signal.aborted) {
            }
        }
    });

    function handleStreamEvent(data) {
        if (data.type === 'error') {
            console.error('API Error:', data.message);
            document.getElementById('test-content').innerHTML = `<p style="color: red; text-align: center;">Error: ${data.message}</p>`;
            document.getElementById('loading').style.display = 'none';
        }
        else if (data.type === 'status') {
            let counter = document.getElementById('q-counter');
            if (counter) {
                counter.style.display = 'block';
                counter.innerHTML = `Анализ<br>теста`;
            }
        }
        else if (data.type === 'init' || data.type === 'progress') {
            updateProgressUI(data.ready || 0, data.total);
        }
        else if (data.type === 'result') {
            document.getElementById('loading').style.display = 'none';
            cleanupProgressUI();
            if (data.answers) {
                renderTestAnswers(data.answers);
            }
        }
    }

    function updateProgressUI(ready, total) {
        let tracker = document.getElementById('dl-progress');
        const loadingSpinner = document.getElementById('loading');

        if (!tracker) {
            tracker = document.createElement('div');
            tracker.id = 'dl-progress';
            tracker.className = 'stealth-progress-inline';
            if (loadingSpinner && loadingSpinner.parentNode) {
                loadingSpinner.parentNode.insertBefore(tracker, loadingSpinner);
            } else {
                document.body.appendChild(tracker);
            }
        }

        let counter = document.getElementById('q-counter');
        if (counter) {
            counter.style.display = 'block';
            if (ready > 0) {
                counter.innerHTML = `Найдено<br><b style="color:#555; font-size:14px; display:block; margin-top:2px;">${total}</b><hr style="margin:6px 10%; border:none; border-top:1px solid #ddd;">Решено<br><b style="color:#555; font-size:14px; display:block; margin-top:2px;">${ready}</b>`;
            } else {
                counter.innerHTML = `Найдено<br><b style="color:#555; display:block; font-size:14px; margin-top:2px;">${total}</b>`;
            }
        }
    }

    function cleanupProgressUI() {
        let counter = document.getElementById('q-counter');
        if (counter) counter.style.display = 'none';
        const tracker = document.getElementById('dl-progress');
        if (tracker) tracker.remove();
    }

    function createQuestionBlock(item, globalIdx) {
        const block = document.createElement('div');
        block.className = 'question-block';

        const title = document.createElement('div');
        title.className = 'question-title';
        // Remove any leading numbers like "15." or "15)" from the question text itself
        const cleanQuestionText = (item.question || '').replace(/^\s*\d+[\.\)]?\s*/, '');
        title.innerText = `${globalIdx + 1}. ${cleanQuestionText}`;
        block.appendChild(title);

        // Support both single correct_index (int) and multiple (array)
        const correctIndices = Array.isArray(item.correct_index)
            ? item.correct_index
            : [item.correct_index];

        if (item.options && Array.isArray(item.options)) {
            item.options.forEach((opt, optIdx) => {
                const isCorrect = correctIndices.includes(optIdx);
                const optDiv = document.createElement('div');
                optDiv.className = 'option';

                if (isCorrect) {
                    optDiv.innerHTML = `
                        <img src="/static/accets/markedbox.svg" class="checkbox-img" alt="Correct">
                        <span class="option-text">${escapeHTML(opt)}</span>
                    `;
                } else {
                    optDiv.innerHTML = `
                        <img src="/static/accets/checkbox.svg" class="checkbox-img" alt="Option">
                        <span class="option-text">${escapeHTML(opt)}</span>
                    `;
                }
                block.appendChild(optDiv);
            });
        }

        return block;
    }

    function renderTestAnswers(answers) {
        const viewer = document.querySelector('.document-viewer');
        const headerHTML = viewer.querySelector('.test-header').outerHTML;
        viewer.innerHTML = '';

        // Max content height per page (1350px page - 60px top padding - 60px bottom padding - ~190px header+divider+margins)
        const MAX_CONTENT_HEIGHT = 1040;

        // Step 1: Create a hidden measuring container with same styles as pdf-page
        const measurePage = document.createElement('div');
        measurePage.className = 'pdf-page';
        measurePage.style.position = 'absolute';
        measurePage.style.visibility = 'hidden';
        measurePage.style.left = '-9999px';
        measurePage.innerHTML = headerHTML + '<div class="divider"></div>';
        const measureContent = document.createElement('div');
        measurePage.appendChild(measureContent);
        viewer.appendChild(measurePage);

        // Step 2: Render all question blocks into the hidden container to measure their heights
        const allBlocks = [];
        answers.forEach((item, idx) => {
            const block = createQuestionBlock(item, idx);
            measureContent.appendChild(block);
            // Force layout
            const height = block.getBoundingClientRect().height + 40; // +40px for margin-bottom
            allBlocks.push({ block, height, item, idx });
        });

        // Remove the measuring container
        viewer.removeChild(measurePage);

        // Step 3: Distribute blocks across pages based on measured heights
        let pages = [];
        let currentPageBlocks = [];
        let currentHeight = 0;

        allBlocks.forEach((entry) => {
            if (currentHeight + entry.height > MAX_CONTENT_HEIGHT && currentPageBlocks.length > 0) {
                // This block doesn't fit — start a new page
                pages.push(currentPageBlocks);
                currentPageBlocks = [];
                currentHeight = 0;
            }
            currentPageBlocks.push(entry);
            currentHeight += entry.height;
        });

        // Push the last page
        if (currentPageBlocks.length > 0) {
            pages.push(currentPageBlocks);
        }

        // Step 4: Build the actual pages
        pages.forEach((pageBlocks) => {
            const pageDiv = document.createElement('div');
            pageDiv.className = 'pdf-page';
            pageDiv.innerHTML = headerHTML + '<div class="divider"></div>';

            const contentDiv = document.createElement('div');

            pageBlocks.forEach((entry, idx) => {
                const block = createQuestionBlock(entry.item, entry.idx);

                const hr = document.createElement('div');
                hr.style.marginTop = '40px';
                hr.style.marginBottom = '10px';
                hr.style.height = '6px';
                hr.style.backgroundColor = '#000';
                block.appendChild(hr);

                contentDiv.appendChild(block);
            });

            const finalHr = document.createElement('div');
            finalHr.className = 'divider';
            finalHr.style.marginTop = '40px';
            finalHr.style.height = '4px';
            contentDiv.appendChild(finalHr);

            pageDiv.appendChild(contentDiv);
            viewer.appendChild(pageDiv);
        });

        // Apply any active zoom to the newly created pages
        applyZoom();
    }

    function escapeHTML(text) {
        if (!text) return '';
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, function (m) { return map[m]; });
    }

    // ========== AUTOMATIC CLASS SCHEDULE TIMER ==========
    // Class end times (hours, minutes)
    const CLASS_SCHEDULE = [
        { h: 10, m: 35 },  // 1 пара
        { h: 12, m: 25 },  // 2 пара
        { h: 14, m: 15 },  // 3 пара
        { h: 16, m: 5 },  // 4 пара
        { h: 17, m: 55 },  // 5 пара
        { h: 19, m: 45 },  // 6 пара
        { h: 21, m: 35 },  // 7 пара
    ];
    const GRACE_PERIOD_MS = 5 * 60 * 1000; // 5 min after class ends — show 00:00
    const RED_THRESHOLD_MS = 5 * 60 * 1000; // last 5 min — red timer

    function getClassEndDates(now) {
        // Build Date objects for today's schedule
        return CLASS_SCHEDULE.map(s =>
            new Date(now.getFullYear(), now.getMonth(), now.getDate(), s.h, s.m, 0)
        );
    }

    function getTimerState(now) {
        const ends = getClassEndDates(now);

        // Check if we're in the 5-min grace period after any class ended
        for (const end of ends) {
            const elapsed = now - end;
            if (elapsed >= 0 && elapsed < GRACE_PERIOD_MS) {
                return { remaining: 0, isGrace: true, isRed: true };
            }
        }

        // Find the next upcoming class end
        for (const end of ends) {
            const diff = end - now;
            if (diff > 0) {
                return {
                    remaining: diff,
                    isGrace: false,
                    isRed: diff <= RED_THRESHOLD_MS
                };
            }
        }

        // All classes for today are over (past last grace period too)
        return { remaining: 0, isGrace: false, isRed: false, allDone: true };
    }

    function updateAllTimers() {
        const now = new Date();
        const state = getTimerState(now);

        let display;
        let color;

        if (state.allDone) {
            display = '--:--';
            color = '#000000';
        } else if (state.isGrace || state.remaining === 0) {
            display = '00:00';
            color = '#cc0000';
        } else {
            const totalMinutes = Math.ceil(state.remaining / 60000); // round up
            const displayHours = Math.floor(totalMinutes / 60);
            const displayMinutes = totalMinutes % 60;
            display = String(displayHours).padStart(2, '0') + ':' + String(displayMinutes).padStart(2, '0');
            color = state.isRed ? '#cc0000' : '#000000';
        }

        document.querySelectorAll('.timer-display').forEach(el => {
            el.textContent = display;
            el.style.color = color;
        });
    }

    // Start the timer immediately and update every second
    updateAllTimers();
    setInterval(updateAllTimers, 1000);



    // Real-time character limit for name fields
    document.body.addEventListener('input', (e) => {
        if (e.target.classList.contains('editable-name')) {
            const text = e.target.textContent;
            if (text.length > 10) {
                e.target.textContent = text.slice(0, 10);
                // Move cursor to end
                const range = document.createRange();
                const sel = window.getSelection();
                range.selectNodeContents(e.target);
                range.collapse(false);
                sel.removeAllRanges();
                sel.addRange(range);
            }
        }
    });

    // Editable name and SRN logic
    document.body.addEventListener('keydown', (e) => {
        if ((e.target.classList.contains('editable-name') || e.target.classList.contains('editable-srn')) && e.key === 'Enter') {
            e.preventDefault();
            e.target.blur();
        }
    });

    document.body.addEventListener('focusout', (e) => {
        if (e.target.classList.contains('editable-name')) {
            let text = e.target.textContent.trim();
            if (text.length > 10) text = text.slice(0, 10);
            if (text.length > 0) {
                text = text.charAt(0).toUpperCase() + text.slice(1).toLowerCase();
            } else {
                text = e.target.id === 'edit-lastname' ? 'Khaverko' : 'Vyacheslav';
            }
            e.target.textContent = text;

            // Sync all other pages
            document.querySelectorAll('#' + e.target.id).forEach(el => {
                if (el !== e.target) el.textContent = text;
            });
        }
        else if (e.target.classList.contains('editable-srn')) {
            let text = e.target.textContent.trim();
            // Limit to 6 characters
            if (text.length > 6) {
                text = text.slice(0, 6);
            }
            if (text.length === 0) text = '240715'; // default SRN

            e.target.textContent = text;

            // Sync all other pages
            document.querySelectorAll('.editable-srn').forEach(el => {
                if (el !== e.target) el.textContent = text;
            });
        }
    });

    // Welcome message is rendered via index.html
});
