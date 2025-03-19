// main.js - Logique principale de l'application

// Variables globales
let isModalOpen = false;
let currentAssetFocus = null;

// Initialisation de l'application
function initApp() {
    console.log("Initialisation de l'application TradingHybrid");
    
    // Initialiser les composants d'interface
    initInterfaceComponents();
    
    // Initialiser les écouteurs d'événements
    setupEventListeners();
    
    // Animation de chargement initial
    animateInterfaceElements();
}

// Initialisation des composants d'interface
function initInterfaceComponents() {
    // Mettre à jour la date du copyright
    updateCopyrightYear();
    
    // Initialiser les cartes des actifs
    initAssetCards();
    
    // Initialiser les modales
    initModals();
}

// Configuration des écouteurs d'événements
function setupEventListeners() {
    // Événements de navigation
    document.getElementById('learnMoreBtn').addEventListener('click', scrollToFeatures);
    document.getElementById('whitepaperBtn').addEventListener('click', openWhitepaper);
    
    // Événements des modales
    document.getElementById('closeWalletModal').addEventListener('click', closeModal);
    
    // Événements des wallets
    document.getElementById('metamaskBtn').addEventListener('click', () => connectWithProvider('metamask'));
    document.getElementById('walletConnectBtn').addEventListener('click', () => connectWithProvider('walletconnect'));
    
    // Événements des assets
    const assetItems = document.querySelectorAll('.asset-item');
    assetItems.forEach(item => {
        item.addEventListener('click', () => focusAsset(item.dataset.symbol));
    });
    
    // Événements du document
    document.addEventListener('click', handleOutsideClick);
    
    // Événements de scroll
    window.addEventListener('scroll', handleScroll);
    
    console.log("Écouteurs d'événements configurés");
}

// Fonctions de navigation
function scrollToFeatures() {
    const featuresSection = document.querySelector('.features');
    featuresSection.scrollIntoView({ behavior: 'smooth' });
}

function openWhitepaper() {
    // Ouvrir le whitepaper dans un nouvel onglet (remplacer par une URL réelle)
    window.open('https://example.com/whitepaper.pdf', '_blank');
}

// Gestion des modales
function initModals() {
    const modals = document.querySelectorAll('.modal');
    modals.forEach(modal => {
        modal.style.display = 'none';
    });
}

function openModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.style.display = 'flex';
        isModalOpen = true;
    }
}

function closeModal() {
    const modals = document.querySelectorAll('.modal');
    modals.forEach(modal => {
        modal.style.display = 'none';
    });
    isModalOpen = false;
}

function handleOutsideClick(event) {
    if (isModalOpen) {
        const modals = document.querySelectorAll('.modal');
        modals.forEach(modal => {
            if (event.target === modal) {
                closeModal();
            }
        });
    }
}

// Gestion de la connexion du wallet
function connectWithProvider(provider) {
    console.log(`Tentative de connexion avec ${provider}`);
    
    if (provider === 'metamask') {
        // La fonction connectWallet() est définie dans web3.js
        connectWallet();
    } else if (provider === 'walletconnect') {
        // Implémenter la connexion WalletConnect
        console.log("WalletConnect n'est pas encore implémenté");
        alert("WalletConnect sera disponible prochainement.");
    }
    
    closeModal();
}

// Fonctions pour les actifs
function initAssetCards() {
    // Animation d'entrée pour les cartes d'actifs
    const assetItems = document.querySelectorAll('.asset-item');
    assetItems.forEach((item, index) => {
        item.style.opacity = '0';
        item.style.transform = 'translateY(20px)';
        setTimeout(() => {
            item.style.transition = 'all 0.3s ease';
            item.style.opacity = '1';
            item.style.transform = 'translateY(0)';
        }, 100 * index);
    });
}

function focusAsset(symbol) {
    console.log(`Focus sur l'actif: ${symbol}`);
    
    // Réinitialiser la classe active
    const assetItems = document.querySelectorAll('.asset-item');
    assetItems.forEach(item => {
        item.classList.remove('active');
    });
    
    // Ajouter la classe active à l'élément sélectionné
    const selectedAsset = document.querySelector(`.asset-item[data-symbol="${symbol}"]`);
    if (selectedAsset) {
        selectedAsset.classList.add('active');
        currentAssetFocus = symbol;
        
        // Simuler le chargement des données pour cet actif
        loadAssetData(symbol);
    }
}

function loadAssetData(symbol) {
    // Simuler un chargement de données
    console.log(`Chargement des données pour ${symbol}...`);
    
    // Cette fonction pourrait être étendue pour charger des données réelles
    // depuis une API ou la blockchain
}

// Animations et effets visuels
function animateInterfaceElements() {
    // Animation d'entrée pour les cartes de fonctionnalités
    const featureCards = document.querySelectorAll('.feature-card');
    featureCards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(30px)';
        setTimeout(() => {
            card.style.transition = 'all 0.5s ease';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, 200 * index);
    });
    
    // Animation des métriques
    const metricBoxes = document.querySelectorAll('.metric-box');
    metricBoxes.forEach((box, index) => {
        box.style.opacity = '0';
        setTimeout(() => {
            box.style.transition = 'opacity 0.5s ease';
            box.style.opacity = '1';
        }, 300 * index);
    });
}

function handleScroll() {
    // Ajouter des effets au défilement
    const scrollPosition = window.scrollY;
    
    // Effet parallaxe sur la section hero
    const heroSection = document.querySelector('.hero');
    if (heroSection) {
        const offset = scrollPosition * 0.3;
        heroSection.style.backgroundPosition = `50% ${offset}px`;
    }
    
    // Animation à l'entrée des sections
    const sections = document.querySelectorAll('section');
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.offsetHeight;
        const windowHeight = window.innerHeight;
        
        if (scrollPosition > sectionTop - windowHeight + sectionHeight / 3) {
            section.classList.add('visible');
        }
    });
}

// Fonctions utilitaires
function updateCopyrightYear() {
    const currentYear = new Date().getFullYear();
    const copyrightElement = document.querySelector('.copyright p');
    if (copyrightElement) {
        copyrightElement.textContent = `© ${currentYear} Trading Hybrid Project. Tous droits réservés.`;
    }
}

// Initialisation au chargement de la page
document.addEventListener('DOMContentLoaded', function() {
    initApp();
    
    // Afficher un message de bienvenue dans la console
    console.log("%cTrading Hybrid Platform", "color: #3498db; font-size: 24px; font-weight: bold;");
    console.log("%cPlateforme de Trading Algorithmique avec Transparence Blockchain", "color: #2ecc71; font-size: 14px;");
});

// Ajouter quelques styles CSS dynamiques
function addDynamicStyles() {
    const style = document.createElement('style');
    style.textContent = `
        .asset-item.active {
            border: 2px solid var(--primary-color);
            transform: scale(1.05);
        }
        
        section.visible .feature-card,
        section.visible .metric-box,
        section.visible .verify-container {
            opacity: 1;
            transform: translateY(0);
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .btn-connect:hover {
            animation: pulse 1s infinite;
        }
    `;
    document.head.appendChild(style);
}

// Ajouter les styles dynamiques
addDynamicStyles();