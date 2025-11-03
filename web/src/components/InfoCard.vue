<template>
  <div class="info-section">
    <div class="card">
      <div class="card-content">
        <h2 class="section-title">Why choose Repo Agent?</h2>
        <p class="card-text">
          Paste a link, and we'll generate<br />detailed wiki for you. Analyze<br />a repository in
          minutes.
        </p>
      </div>
      <div class="card-icon">
        <div class="gradient-blob"></div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.info-section {
  width: 100%;
  max-width: 450px;
  margin: 0 auto;
}

.section-title {
  font-size: 16px;
  font-weight: 600;
  margin: 0 0 16px 0;
  color: var(--title-color);
  text-align: left;
  transition: color 0.3s;
}

.card {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 32px;
  background: var(--card-bg, white);
  border-radius: 10px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
  gap: 32px;
  transition: background-color 0.3s;
}

.card-content {
  flex: 1;
}

.card-text {
  font-size: 14px;
  line-height: 1.6;
  color: var(--secondary-text);
  margin: 0;
  transition: color 0.3s;
}

.card-icon {
  width: 100px;
  height: 100px;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
}

.gradient-blob {
  width: 100%;
  height: 100%;
  border-radius: 50%;
  background:
    radial-gradient(circle at 70% 30%, #f56565 0%, rgba(245, 101, 101, 0) 40%),
    radial-gradient(circle at 30% 70%, #667eea 0%, rgba(102, 126, 234, 0) 50%);
  filter: blur(20px);
  opacity: 0.8;
}

@media (max-width: 640px) {
  .card {
    flex-direction: column;
    padding: 32px 24px;
    gap: 24px;
  }

  .card-text {
    text-align: center;
  }

  .card-icon {
    width: 100px;
    height: 100px;
  }
}
</style>
