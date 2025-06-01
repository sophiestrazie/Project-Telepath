
# 📁 DATA_INSTRUCTIONS.md – How to Access the Algonauts 2025 Data

This project uses the **Algonauts 2025 dataset**, which contains large movie and fMRI files. Due to the size of the dataset, we **do not upload data files to GitHub**. Instead, all collaborators must **download the data locally** using [DataLad](https://www.datalad.org/), a tool designed to manage large scientific datasets efficiently.

---

## ⚙️ Step-by-Step Instructions

### 1. ✅ Install DataLad

If you don’t already have it installed:

```bash
pip install datalad
```

Or using `conda`:

```bash
conda install -c conda-forge datalad
```

---

### 2. 📥 Clone the Algonauts Dataset

Open your terminal and run:

```bash
datalad clone https://github.com/algonauts2025/algonauts2025.git algonauts_2025
cd algonauts_2025
```

This creates a lightweight copy of the repository.

---

### 3. 📦 Download the Full Dataset

From within the cloned folder, run:

```bash
datalad get -r -J8 .
```

- `-r` = recursively get all subfolders
- `-J8` = download with 8 parallel jobs for speed

💡 You can also run `datalad get` on specific folders (e.g. `stimuli/` or `participants/`) if you don’t need everything.

---

### 4. 🧠 Use the Data in Your Scripts

All scripts in this project should reference the data **as local file paths** like:

```python
"../algonauts_2025/stimuli/movies/friends/s2/friends_s02e13b.mkv"
```

Make sure your local folder names match this structure.

---

## 📂 Recommended Folder Structure

```
your_project/
│
├── notebooks/
│   └── analysis.ipynb
├── src/
│   └── processing.py
├── algonauts_2025/          ← Cloned & downloaded via DataLad (not tracked in Git)
├── .gitignore
├── README.md
└── DATA_INSTRUCTIONS.md     ← You're reading this
```

---

## 🛡️ Important: Do Not Upload Data to GitHub

Add the following to your `.gitignore`:

```
algonauts_2025/
*.mkv
*.h5
*.nii
```

This prevents large files from being pushed to GitHub.

---

## ❓ Need Help?

If you’re having trouble with setup, feel free to:
- Post in the team chat
- Create a GitHub Issue
- Ask Maria or the technical lead
