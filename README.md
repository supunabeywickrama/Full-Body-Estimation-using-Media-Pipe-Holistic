# Full-Body-Estimation-using-Media-Pipe-Holistic

# 01.  RehabX – Finger Runner Game  
*A Vision-Based Rehabilitation Game Using MediaPipe & Pygame*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic-green)
![Pygame](https://img.shields.io/badge/Pygame-2.x-orange)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-yellow)

---


## 🎮 Demo Video

<p align="center">
  <a href="https://drive.google.com/file/d/18xjtzSG73Yo-BCUROskW4zco3F5WXFkE/view?usp=drive_link" target="_blank">
    <button style="padding:10px 18px; font-size:16px; cursor:pointer;">
      ▶ Watch Demo Video
    </button>
  </a>
</p>




---

## 🧠 Project Overview

**RehabX – Finger Runner** is an interactive **gesture-controlled rehabilitation game** designed for **hand and finger motor recovery**.

Using a standard webcam and **MediaPipe Hands**, the system:
- Tracks **individual finger ability**
- Converts finger motion into **game controls**
- Motivates patients through **gamified feedback, scores, and achievements**

No wearables. No controllers. Just your hand.

---

## ✋ Core Features

### 🔍 Hand & Finger Tracking
- Real-time hand detection using **MediaPipe**
- Automatic **left / right hand correction**
- Finger curl estimation per finger

### 🎯 Manual Finger Calibration
- Step-by-step guided calibration  
  - Thumb → Index → Middle → Ring → Pinky
- Capture **0% (rest)** and **100% (full flex)** per finger
- Calibration profiles saved per hand
- Reuse calibration on next launch

### 🕹️ Gesture-Based Gameplay
| Finger | Action |
|------|------|
| Thumb | Jump (height based on flex %) |
| Index | Move right |
| Middle | Move left |
| Ring | Shield |
| Pinky | Brake / slow down |

### 🏃 Endless Runner Game
- Obstacle avoidance
- Variable jump height
- Speed progression
- Combo & smoothness scoring
- Session logging for rehab analysis

### 🏅 Motivation & Feedback
- Score & combo system
- Smooth-control bonus
- Target-hold rehab rewards
- Visual feedback & particle effects

---

## 🧪 Designed For

- 🏥 Stroke rehabilitation  
- ✋ Hand / finger motor recovery  
- 🧠 Human-computer interaction research  
- 🎓 Academic & final-year projects  

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **MediaPipe Hands**
- **OpenCV**
- **Pygame**
- **NumPy**

---

## 🚀 How to Run

### 1️⃣ Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/RehabX-Finger-Runner.git
cd RehabX-Finger-Runner
