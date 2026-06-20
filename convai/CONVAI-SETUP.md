# Convai Setup Guide — SimXLabs Simulation Concierge

Step-by-step to get the Convai character live and connected to the SimXLabs API.

---

## Before you start

Make sure:
- [ ] SimXLabs pilot API is running (see main README)
- [ ] You have the public API URL (ngrok URL for local, or deployed URL)
- [ ] You're logged into Convai at convai.com

---

## Step 1 — Confirm External API access

1. Go to **convai.com → Settings → Plan**
2. Check if "External API" is listed as an available feature
3. If not: upgrade to Professional, or email team@convai.com referencing the SimXLabs pilot

---

## Step 2 — Create the character

1. Go to **Playground → + New Character**
2. Name: `Simra`
3. Voice: Pick a calm, professional voice (suggest: "Sarah" or "Nova" if available)
4. Paste the full contents of `character-system-prompt.md` into the **Backstory** field

---

## Step 3 — Upload the Knowledge Bank

1. In the character editor, go to **Knowledge Bank**
2. Click **Upload Document**
3. Upload `knowledge-bank.md`
4. Wait for status to show **"Processed"** (takes ~1–2 minutes)
5. Click **Connect to Character**

---

## Step 4 — Configure External API methods

Repeat for each of the 3 methods in `external-api-methods.json`:

1. In the character editor, go to **External API**
2. Click **+ Add Method**
3. Fill in:
   - **Name**: (e.g. `start_simulation`)
   - **Description**: (copy from the JSON)
   - **HTTP Method**: POST or GET (as specified)
   - **Endpoint URL**: replace `YOUR_API_URL` with your actual API URL
   - **Parameters**: add each one with its name, type, and description
4. Click **Test** to verify it connects
5. Save

---

## Step 5 — Test in Playground

Open the Convai Playground with your character and try:

> "I need 10,000 bin picking demonstrations with high diversity"

Expected flow:
1. Simra triggers `start_simulation` and tells you the run ID
2. Ask: "How's that run going?"
3. Simra triggers `check_run_status` and reports progress
4. Ask: "Is it done yet?" or wait ~90 seconds
5. Simra triggers `get_run_summary` and narrates full results + trace link

---

## Step 6 — Publish (optional)

**Avatar Studio (recommended for demos):**
1. In character editor → **Experiences → Avatar Studio**
2. Customize avatar appearance
3. Click **Publish**
4. Copy the shareable link

**Convai Sim (3D walkable environment):**
1. Go to **Convai Sim → New Experience**
2. Choose an environment (industrial/lab setting works best)
3. Add your Simra character
4. Publish → copy embed code or link

---

## Troubleshooting

**External API calls not working:**
- Confirm the API URL is publicly accessible (not `localhost`)
- Use [ngrok](https://ngrok.com) for local testing: `ngrok http 8000`
- Check that the endpoint matches exactly (trailing slashes matter)

**High response latency (>5s):**
- Try a different model in Convai Core AI Settings
- Keep the Knowledge Bank document under 5,000 words
- Reduce system prompt length if needed

**Character doesn't trigger actions:**
- Make sure the action descriptions are specific and include example trigger phrases
- Add "When the user asks X, call Y" explicitly in the system prompt

---

## Sharing the demo

Once published, share the Avatar Studio link directly:
`https://convai.com/experience/{your-experience-id}`

No installation needed — runs in browser. Include this in your Techstars demo and OEM outreach.
