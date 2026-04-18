export const detectionService = {
  async detect(data: Record<string, number>) {
    // Simulate model inference
    const response = await fetch('/api/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    })
    return response.json()
  },
}