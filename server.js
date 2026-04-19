const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');

const server = http.createServer((req, res) => {
    const parsedUrl = url.parse(req.url, true);
    const pathname = parsedUrl.pathname;

    // Simple in-memory cache for last row counts per client/file
    const rowCache = new Map();

    // Game CSV append endpoint POST /append-game-csv
    if (req.method === 'POST' && pathname === '/append-game-csv') {
        let body = '';
        req.on('data', chunk => { body += chunk; });
        req.on('end', () => {
            try {
                const data = JSON.parse(body);
                const filePath = path.join(__dirname, data.filename);
                
                const csvRows = Array.isArray(data.rows) ? data.rows : [data.rows];
                const content = csvRows.join('\n') + '\n';
                
                fs.appendFile(filePath, content, (err) => {
                    if (err) {
                        res.writeHead(500);
                        res.end(JSON.stringify({error: 'Append failed'}));
                        return;
                    }
                    res.writeHead(200);
                    res.end(JSON.stringify({success: true}));
                });
            } catch (e) {
                res.writeHead(400);
                res.end(JSON.stringify({error: 'Invalid JSON'}));
            }
        });
        return;
    }

    // Special endpoint to list CSV files (include game CSVs)
    if (pathname === '/csv-files') {
        fs.readdir(__dirname, (err, files) => {
            if (err) {
                res.writeHead(500);
                res.end('Internal server error');
                return;
            }

            const csvFiles = files
                .filter(file => file.endsWith('.csv') && (file.startsWith('rehab_session_') || file.startsWith('game_rehab_session_')))
                .sort()
                .reverse();

            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(csvFiles));
        });
        return;
    }

    // Incremental CSV endpoint: /incremental/filename.csv?lastRow=123
    if (pathname.startsWith('/incremental/')) {
        const filename = decodeURIComponent(pathname.slice(12)); // Remove /incremental/
        const lastRow = parseInt(parsedUrl.query.lastRow) || 0;
        const filePath = path.join(__dirname, filename);

        fs.access(filePath, fs.constants.F_OK, (err) => {
            if (err) {
                res.writeHead(404, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({error: 'File not found', newRows: [], totalRows: 0}));
                return;
            }

            fs.readFile(filePath, 'utf8', (err, csvContent) => {
                if (err) {
                    res.writeHead(500, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({error: 'Read error', newRows: [], totalRows: 0}));
                    return;
                }

                // Parse CSV to get rows (simple split for demo, use csv-parser in prod)
                const lines = csvContent.trim().split('\n');
                const totalRows = lines.length;
                const newRows = [];

                if (lastRow < totalRows) {
                    // Send header + new rows, or just new rows if lastRow > 0
                    if (lastRow === 0) {
                        newRows.push(lines[0]); // Include header on first request
                    }
                    for (let i = Math.max(1, lastRow); i < totalRows; i++) {
                        newRows.push(lines[i]);
                    }
                }

                const response = {
                    newRows: newRows.length,
                    newDataPoints: newRows.slice(-10), // Last 10 for preview
                    totalRows: totalRows,
                    lastRow: totalRows
                };

                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify(response));
            });
        });
        return;
    }

    let filePath = path.join(__dirname, pathname === '/' ? 'index.html' : pathname);
    
    // Handle root request
    if (pathname === '/') {
        filePath = path.join(__dirname, 'index.html');
    }
    
    // Check if file exists
    fs.access(filePath, fs.constants.F_OK, (err) => {
        if (err) {
            res.writeHead(404);
            res.end('File not found');
            return;
        }
        
        // Read and serve the file
        fs.readFile(filePath, (err, data) => {
            if (err) {
                res.writeHead(500);
                res.end('Internal server error');
                return;
            }
            
            // Set content type based on file extension
            const ext = path.extname(filePath);
            let contentType = 'text/html';
            switch (ext) {
                case '.js':
                    contentType = 'text/javascript';
                    break;
                case '.css':
                    contentType = 'text/css';
                    break;
                case '.json':
                    contentType = 'application/json';
                    break;
                case '.png':
                    contentType = 'image/png';
                    break;
                case '.jpg':
                case '.jpeg':
                    contentType = 'image/jpeg';
                    break;
                case '.gif':
                    contentType = 'image/gif';
                    break;
                case '.wav':
                    contentType = 'audio/wav';
                    break;
                case '.mp3':
                    contentType = 'audio/mpeg';
                    break;
                case '.svg':
                    contentType = 'image/svg+xml';
                    break;
                case '.ico':
                    contentType = 'image/x-icon';
                    break;
                case '.csv':
                    contentType = 'text/csv';
                    break;
            }
            
            res.writeHead(200, { 'Content-Type': contentType });
            res.end(data);
        });
    });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});