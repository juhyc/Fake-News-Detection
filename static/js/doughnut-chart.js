var prob = {{label_probs|tojson}};

new Chart(document.getElementById("doughnut-chart"), {
    type: 'doughnut',
    data: {
      labels: ["Real News", "Fake News"],
      datasets: [
        {
          label: "",
          backgroundColor: ["#8e5ea2","#e8c3b9"],
          data: prob
    }
    }
        }
      ]
    },
    options: {
      title: {
        display: true,
        text: 'Check your News'
      }
    }
});