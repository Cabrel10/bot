// charts.js - Gestion des visualisations et graphiques

// Variables globales
let performanceChart;

// Initialisation des graphiques
function initCharts() {
  const ctx = document.getElementById('performanceChart').getContext('2d');
  
  // Configuration du graphique de performance
  performanceChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
      datasets: [{
        label: 'Performance ROI',
        data: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        backgroundColor: 'rgba(52, 152, 219, 0.2)',
        borderColor: 'rgba(52, 152, 219, 1)',
        borderWidth: 2,
        tension: 0.4,
        pointBackgroundColor: 'rgba(52, 152, 219, 1)',
        pointBorderColor: '#fff',
        pointRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: false,
          grid: {
            color: 'rgba(255, 255, 255, 0.1)',
          },
          ticks: {
            color: 'rgba(255, 255, 255, 0.7)',
            callback: function(value) {
              return value + '%';
            }
          }
        },
        x: {
          grid: {
            color: 'rgba(255, 255, 255, 0.1)',
          },
          ticks: {
            color: 'rgba(255, 255, 255, 0.7)'
          }
        }
      },
      plugins: {
        legend: {
          labels: {
            color: 'rgba(255, 255, 255, 0.7)'
          }
        },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.9)',
          titleColor: 'rgba(255, 255, 255, 0.9)',
          bodyColor: 'rgba(255, 255, 255, 0.7)',
          borderColor: 'rgba(52, 152, 219, 0.5)',
          borderWidth: 1,
          callbacks: {
            label: function(context) {
              return `Performance: ${context.parsed.y}%`;
            }
          }
        }
      }
    }
  });
  
  console.log("Graphiques initialisés");
}

// Mise à jour du graphique de performance
function updatePerformanceChart(performanceData) {
  // Génération de données pour la démonstration
  // Dans un cas réel, ces données proviendraient de la blockchain
  const monthlyData = generateMonthlyPerformanceData(performanceData);
  
  // Mise à jour des datasets
  performanceChart.data.datasets[0].data = monthlyData;
  
  // Mise à jour des couleurs en fonction des valeurs
  updateChartColors(performanceChart, monthlyData);
  
  // Rafraîchir le graphique
  performanceChart.update();
  
  console.log("Graphique de performance mis à jour");
}

// Génération de données de performance mensuelles (demo)
function generateMonthlyPerformanceData(performanceData) {
  // Générer des données de performance pour la démonstration
  // Celles-ci seraient normalement récupérées depuis la blockchain
  const baseROI = performanceData ? (performanceData.roi / 100) : 3.5;
  
  const monthlyData = [];
  let cumulative = 0;
  
  for (let i = 0; i < 12; i++) {
    // Générer une variation aléatoire autour de la valeur de base
    const monthVariation = (Math.random() * 2 - 0.5) * baseROI;
    const monthROI = baseROI + monthVariation;
    cumulative += monthROI;
    monthlyData.push(cumulative.toFixed(2));
  }
  
  return monthlyData;
}

// Mise à jour des couleurs du graphique
function updateChartColors(chart, data) {
  // Définir la couleur en fonction de la tendance (positif/négatif)
  const lastValue = parseFloat(data[data.length - 1]);
  
  if (lastValue > 0) {
    chart.data.datasets[0].borderColor = 'rgba(46, 204, 113, 1)';
    chart.data.datasets[0].backgroundColor = 'rgba(46, 204, 113, 0.2)';
    chart.data.datasets[0].pointBackgroundColor = 'rgba(46, 204, 113, 1)';
  } else {
    chart.data.datasets[0].borderColor = 'rgba(231, 76, 60, 1)';
    chart.data.datasets[0].backgroundColor = 'rgba(231, 76, 60, 0.2)';
    chart.data.datasets[0].pointBackgroundColor = 'rgba(231, 76, 60, 1)';
  }
}

// Initialisation des graphiques au chargement de la page
document.addEventListener('DOMContentLoaded', function() {
  initCharts();
  
  // Données de démonstration initiales
  const demoPerformance = {
    roi: 350, // 3.5%
    winRate: 6200, // 62%
    sharpRatio: 180 // 1.8
  };
  
  // Mise à jour initiale du graphique avec les données de démo
  updatePerformanceChart(demoPerformance);
  
  // Mise à jour des indicateurs de performance avec les données de démo
  document.getElementById('roi30d').textContent = (demoPerformance.roi / 100).toFixed(2) + '%';
  document.getElementById('winRate').textContent = (demoPerformance.winRate / 100).toFixed(2) + '%';
  document.getElementById('sharpRatio').textContent = (demoPerformance.sharpRatio / 100).toFixed(2);
});