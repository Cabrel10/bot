# Guide des erreurs communes et leurs solutions

## Types d'erreurs

### 1. Erreurs d'importation (`ImportError`)

```python
ImportError: cannot import name 'TimeFrame' from 'trading.core.data_types'
```

**Cause**: Cette erreur survient lorsque vous essayez d'importer une classe ou une fonction qui n'existe pas ou a été renommée.

**Solutions**:
- Vérifier l'existence de la classe/fonction dans le module source
- Remplacer l'import par la nouvelle version (par exemple, remplacer `TimeFrame` par les constantes `TIMEFRAME_*`)
- Ajouter la classe/fonction manquante au module

### 2. Erreurs de dataclass (`TypeError`)

```python
TypeError: non-default argument 'symbol' follows default argument
```

**Cause**: Dans les dataclasses Python, les paramètres sans valeur par défaut doivent toujours être placés avant les paramètres avec valeur par défaut.

**Solutions**:
- Réorganiser les champs de la dataclass pour placer les paramètres obligatoires en premier
- Utiliser l'option `kw_only=True` pour les paramètres avec valeur par défaut (Python 3.10+)
- Dans les classes héritées, redéfinir les paramètres optionnels avec `field(kw_only=True)`

### 3. Erreurs d'exécution (`RuntimeError`)

```python
RuntimeError: Model graph not available
```

**Cause**: Le modèle TensorFlow n'a pas été correctement initialisé ou chargé.

**Solutions**:
- Vérifier que le modèle a été correctement chargé
- S'assurer que la version de TensorFlow est compatible
- Désactiver eager execution si nécessaire

### 4. Erreurs de configuration (`ValueError`)

```python
ValueError: Invalid exchange configuration
```

**Cause**: La configuration fournie pour un échange ou un autre composant n'est pas valide.

**Solutions**:
- Vérifier les fichiers de configuration YAML
- S'assurer que les valeurs requises sont présentes
- Valider les types de données des paramètres

### 5. Erreurs d'accès aux données (`FileNotFoundError`)

```python
FileNotFoundError: [Errno 2] No such file or directory: '/path/to/data.db'
```

**Cause**: Le système tente d'accéder à un fichier qui n'existe pas.

**Solutions**:
- Vérifier le chemin du fichier
- Créer les répertoires manquants
- Initialiser les fichiers de données si nécessaire

## Procédure de débogage

### 1. Identifier l'erreur
Lisez attentivement le message d'erreur et la trace d'appel (stack trace) pour déterminer l'origine exacte du problème.

### 2. Isoler le problème
Essayez de reproduire l'erreur avec un exemple minimal, par exemple:
```python
python -c "from trading.core.data_types import *; print('Import successful!')"
```

### 3. Appliquer la solution appropriée
En fonction du type d'erreur, appliquez la solution correspondante décrite ci-dessus.

### 4. Tester la solution
Vérifiez que l'erreur a été résolue en exécutant à nouveau le code problématique.

### 5. Documenter la solution
Ajoutez un commentaire dans le code pour expliquer la solution et éviter de futurs problèmes similaires.

## Erreurs spécifiques au projet

### Erreur de type dataclass pour les paramètres
Les classes héritant de `BaseData` peuvent rencontrer des problèmes lors de la définition de nouveaux champs, car `BaseData` définit déjà `metadata` avec une valeur par défaut.

**Solution**: Modifier `BaseData` pour utiliser `kw_only=True`:
```python
@dataclass
class BaseData:
    metadata: Dict[str, Any] = field(default_factory=dict, kw_only=True)
```

### Erreur d'importation TimeFrame
Les imports de `TimeFrame` doivent être remplacés par les constantes de chaînes de caractères.

**Solution**: Remplacer:
```python
from .data_types import MarketData, OrderData, TradeData, TimeFrame
```
Par:
```python
from .data_types import MarketData, OrderData, TradeData
from .types import (TIMEFRAME_1M, TIMEFRAME_5M, TIMEFRAME_15M, ...)
``` 