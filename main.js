let chatData = {};

// Image Modal Logic
const modal = document.getElementById("imageModal");
const expandedImg = document.getElementById("expandedImg");
document.querySelector(".close-modal").onclick = () => modal.style.display = "none";

function openModal(src) {
    if (!src) return;
    modal.style.display = "block";
    expandedImg.src = src;
}

document.getElementById('folderInput').addEventListener('change', async (e) => {
    const files = e.target.files;
    chatData = {};

    for (let file of files) {
        const parts = file.webkitRelativePath.split('/');
        if (parts.length < 3) continue;
        const chatName = parts[1];
        const fileName = parts[parts.length - 1].toLowerCase();

        if (!chatData[chatName]) chatData[chatName] = { json: null, media: {} };

        if (fileName.endsWith('.json')) {
            chatData[chatName].json = JSON.parse(await file.text());
        } else if (fileName.endsWith('.png') || fileName.endsWith('.svg')) {
            const url = URL.createObjectURL(file);
            if (fileName.includes('word')) chatData[chatName].media.word = url;
            else if (fileName.includes('pie')) chatData[chatName].media.pie = url;
            else if (fileName.includes('per_day')) chatData[chatName].media.perDay = url;
            else if (fileName.includes('day_time')) chatData[chatName].media.dayTime = url;
        }
    }
    renderList();
});

function renderList() {
    const list = document.getElementById('chatList');
    list.innerHTML = '';
    Object.keys(chatData).sort().forEach(name => {
        const li = document.createElement('li');
        li.textContent = name;
        li.onclick = () => displayChat(name, li);
        list.appendChild(li);
    });
}

function displayChat(name, el) {
    document.querySelectorAll('#chatList li').forEach(l => l.classList.remove('active'));
    el.classList.add('active');
    
    const data = chatData[name];
    const j = data.json;
    if (!j) return;

    document.getElementById('welcomeMessage').classList.add('hidden');
    document.getElementById('statsContent').classList.remove('hidden');

    // Header Stats
    document.getElementById('activeChatName').textContent = name;
    document.getElementById('totalMsgs').textContent = j.total_messages.toLocaleString();
    document.getElementById('daysActive').textContent = Math.round(j.days_active);

    // Image Setup
    const setImg = (id, src) => {
        const img = document.getElementById(id);
        img.src = src || '';
        img.onclick = () => openModal(src);
    };
    setImg('wordCloudImg', data.media.word);
    setImg('pieChartImg', data.media.pie);
    setImg('messagePerDayImg', data.media.perDay);
    setImg('daytimeImg', data.media.dayTime);

    // Streaks Data
    document.getElementById('maxStreakValue').textContent = `${j.message_streaks.max_message_streak.length} Days`;
    document.getElementById('maxStreakDates').textContent = `${j.message_streaks.max_message_streak.start} to ${j.message_streaks.max_message_streak.end}`;
    document.getElementById('maxGapValue').textContent = `${j.message_streaks.max_gap.length} Days`;
    document.getElementById('maxGapDates').textContent = `${j.message_streaks.max_gap.start} to ${j.message_streaks.max_gap.end}`;

    // Tables
    fillTable('freqMessages', Object.entries(j.most_frequent_messages).slice(0, 10));
    fillTable('activeDays', j.most_active_days.slice(0, 10).map(d => [d.date, d.messages]));

    // Emojis Logic
    renderEmojiSection('emojisSent', j.most_frequent_emojis_sent);
    renderEmojiSection('emojisReacted', j.most_frequent_emojis_reacted);

    // Avg Length
    const avgDiv = document.getElementById('avgLengthList');
    avgDiv.innerHTML = '';
    Object.entries(j.average_message_length).forEach(([user, len]) => {
        avgDiv.innerHTML += `<div><strong>${user}:</strong> ${len} chars</div>`;
    });
}

function renderEmojiSection(containerId, emojiData) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    Object.entries(emojiData).forEach(([user, emojis]) => {
        if (Object.keys(emojis).length === 0) return;
        const block = document.createElement('div');
        block.className = 'emoji-block';
        block.innerHTML = `<b>${user}</b>`;
        Object.entries(emojis).forEach(([emo, count]) => {
            block.innerHTML += `<span class="emoji-item">${emo} ${count}</span>`;
        });
        container.appendChild(block);
    });
}

function fillTable(id, rows) {
    const tb = document.querySelector(`#${id} tbody`);
    tb.innerHTML = '';
    rows.forEach(r => {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${r[0]}</td><td>${r[1]}</td>`;
        tb.appendChild(tr);
    });
}