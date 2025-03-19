// web3.js - Gestion des interactions avec la blockchain

// Configuration des constantes
const CONTRACT_ADDRESS = '0x123456789abcdef123456789abcdef123456789';
const CONTRACT_ABI = [
  // Remplacez par l'ABI réel de votre contrat
  {
    "inputs": [],
    "name": "getPerformanceData",
    "outputs": [
      {
        "components": [
          { "internalType": "uint256", "name": "timestamp", "type": "uint256" },
          { "internalType": "int256", "name": "roi", "type": "int256" },
          { "internalType": "uint256", "name": "winRate", "type": "uint256" },
          { "internalType": "int256", "name": "sharpRatio", "type": "int256" }
        ],
        "internalType": "struct TradingData.PerformanceData",
        "name": "",
        "type": "tuple"
      }
    ],
    "stateMutability": "view",
    "type": "function"
  }
];

// Variables globales
let web3;
let userAccount;
let contract;
let chainId;

// Initialisation de Web3
async function initWeb3() {
  // Vérifier si MetaMask est installé
  if (window.ethereum) {
    try {
      web3 = new Web3(window.ethereum);
      console.log("Web3 initialisé avec Ethereum provider");
    } catch (error) {
      console.error("Erreur lors de l'initialisation de Web3:", error);
      showError("Impossible d'initialiser Web3. Veuillez vérifier votre navigateur.");
    }
  } else if (window.web3) {
    // Ancien navigateurs
    web3 = new Web3(window.web3.currentProvider);
    console.log("Web3 initialisé avec provider existant");
  } else {
    // Fallback sur un provider infura
    web3 = new Web3(new Web3.providers.HttpProvider("https://mainnet.infura.io/v3/YOUR_INFURA_KEY"));
    console.log("Web3 initialisé avec Infura");
    showError("Aucun wallet détecté. Certaines fonctionnalités pourraient être limitées.");
  }

  // Initialisation du contrat
  try {
    contract = new web3.eth.Contract(CONTRACT_ABI, CONTRACT_ADDRESS);
    console.log("Contrat initialisé");
  } catch (error) {
    console.error("Erreur d'initialisation du contrat:", error);
  }
}

// Connexion du wallet
async function connectWallet() {
  if (!window.ethereum) {
    showWalletModal();
    return;
  }

  try {
    // Demander à l'utilisateur de se connecter
    const accounts = await window.ethereum.request({ method: 'eth_requestAccounts' });
    userAccount = accounts[0];
    chainId = await web3.eth.getChainId();
    
    // Mise à jour de l'interface utilisateur
    document.getElementById('walletAddress').textContent = formatAddress(userAccount);
    document.getElementById('walletInfo').classList.remove('hidden');
    document.getElementById('connectWallet').classList.add('hidden');
    
    // Vérifier le réseau
    updateNetworkStatus(chainId);
    
    // Charger les données
    loadBlockchainData();
    
    console.log("Wallet connecté:", userAccount);
    
    // Écouter les événements de changement de compte
    window.ethereum.on('accountsChanged', handleAccountsChanged);
    window.ethereum.on('chainChanged', handleChainChanged);
  } catch (error) {
    console.error("Erreur de connexion wallet:", error);
    showError("Impossible de se connecter au wallet. Veuillez réessayer.");
  }
}

// Gestion des changements de compte
function handleAccountsChanged(accounts) {
  if (accounts.length === 0) {
    // L'utilisateur s'est déconnecté
    disconnectWallet();
  } else if (accounts[0] !== userAccount) {
    userAccount = accounts[0];
    document.getElementById('walletAddress').textContent = formatAddress(userAccount);
    loadBlockchainData();
  }
}

// Gestion des changements de réseau
function handleChainChanged(chainId) {
  // Recharger la page comme recommandé par MetaMask
  window.location.reload();
}

// Déconnexion du wallet
function disconnectWallet() {
  userAccount = null;
  document.getElementById('walletInfo').classList.add('hidden');
  document.getElementById('connectWallet').classList.remove('hidden');
  document.getElementById('walletAddress').textContent = "Non connecté";
}

// Mise à jour du statut du réseau
function updateNetworkStatus(chainId) {
  const networkStatus = document.getElementById('networkStatus');
  
  // Définir le statut visuel en fonction du réseau
  if (chainId === 1) {
    // Ethereum Mainnet
    networkStatus.style.backgroundColor = "var(--success-color)";
    networkStatus.title = "Ethereum Mainnet";
  } else if (chainId === 137) {
    // Polygon
    networkStatus.style.backgroundColor = "var(--secondary-color)";
    networkStatus.title = "Polygon Mainnet";
  } else if ([3, 4, 5, 42].includes(chainId)) {
    // Testnets
    networkStatus.style.backgroundColor = "var(--warning-color)";
    networkStatus.title = "Testnet";
  } else {
    // Autre réseau
    networkStatus.style.backgroundColor = "var(--error-color)";
    networkStatus.title = "Réseau non supporté";
  }
}

// Chargement des données de la blockchain
async function loadBlockchainData() {
  if (!contract || !userAccount) return;
  
  try {
    // Récupération des données de performance
    const performance = await contract.methods.getPerformanceData().call({ from: userAccount });
    
    // Mise à jour des métriques
    document.getElementById('roi30d').textContent = (performance.roi / 100).toFixed(2) + '%';
    document.getElementById('winRate').textContent = (performance.winRate / 100).toFixed(2) + '%';
    document.getElementById('sharpRatio').textContent = (performance.sharpRatio / 100).toFixed(2);
    
    // Mise à jour du graphique
    updatePerformanceChart(performance);
    
    console.log("Données blockchain chargées avec succès");
  } catch (error) {
    console.error("Erreur lors du chargement des données:", error);
    showError("Impossible de charger les données depuis la blockchain.");
  }
}

// Vérification des résultats sur la blockchain
async function verifyResults() {
  if (!contract || !userAccount) {
    showError("Veuillez connecter votre wallet pour vérifier les résultats.");
    return;
  }
  
  try {
    // Simulation d'une vérification réussie
    document.getElementById('verificationResult').classList.remove('hidden');
    setTimeout(() => {
      document.getElementById('verificationResult').classList.add('hidden');
    }, 5000);
    
    console.log("Vérification des résultats effectuée");
  } catch (error) {
    console.error("Erreur lors de la vérification:", error);
    showError("Échec de la vérification. Veuillez réessayer.");
  }
}

// Ouvrir Etherscan
function openEtherscan() {
  window.open(`https://etherscan.io/address/${CONTRACT_ADDRESS}`, '_blank');
}

// Fonctions utilitaires
function formatAddress(address) {
  return address.substring(0, 6) + '...' + address.substring(address.length - 4);
}

function showError(message) {
  // Implémenter l'affichage d'erreur
  console.error(message);
  // Exemple: alert(message);
}

function showWalletModal() {
  const modal = document.getElementById('walletModal');
  modal.style.display = 'flex';
}

// Initialisation au chargement de la page
document.addEventListener('DOMContentLoaded', function() {
  // Initialiser Web3
  initWeb3();
  
  // Écouteurs d'événements
  document.getElementById('connectWallet').addEventListener('click', connectWallet);
  document.getElementById('viewContractBtn').addEventListener('click', openEtherscan);
  document.getElementById('verifyResultsBtn').addEventListener('click', verifyResults);
  
  // Initialiser l'adresse du contrat
  document.getElementById('contractAddress').textContent = formatAddress(CONTRACT_ADDRESS);
});