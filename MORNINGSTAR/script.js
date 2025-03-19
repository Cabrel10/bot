// Script JavaScript pour le site MORNINGSTAR
console.log('Bienvenue sur le site MORNINGSTAR');

// Ajouter des interactions pour les boutons
const ctaButtons = document.querySelectorAll('.cta-button');
ctaButtons.forEach(button => {
    button.addEventListener('click', () => {
        alert('Merci pour votre intérêt !');
    });
});

// Ajouter une animation pour la barre de progression
const progressBarFills = document.querySelectorAll('.progress-bar-fill');
progressBarFills.forEach(fill => {
    fill.style.width = fill.style.width;
});

// Ajouter des animations d'apparition progressive
const sections = document.querySelectorAll('section');
const options = {
    threshold: 0.1
};
const observer = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('visible');
            observer.unobserve(entry.target);
        }
    });
}, options);
sections.forEach(section => {
    observer.observe(section);
});
