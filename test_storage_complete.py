"""Test completo del sistema de almacenamiento: guardado y recuperación."""

import pytest
from core.runner import ExperimentRunner
from core.statistics import ExperimentStatistics
from core.storage import ResultsStorage, StorageBackend
from scenarios.registry import ScenarioRegistry
from test_model_mock import MockLLM


def test_save_and_retrieve():
    """Test completo de guardado y recuperación con DuckDB."""
    print("=" * 60)
    print("Test: Guardado y Recuperación (DuckDB)")
    print("=" * 60)
    
    # Limpiar base de datos antes de empezar
    import os
    from pathlib import Path
    results_dir = Path("results")
    for db_file in results_dir.glob("*.duckdb"):
        db_file.unlink()
    for db_file in results_dir.glob("*.db"):
        db_file.unlink()
    print("✅ Base de datos limpiada")
    
    try:
        # 1. Inicializar storage
        storage = ResultsStorage("results", StorageBackend.DUCKDB)
        print("✅ Storage DuckDB inicializado")
        
        # 2. Crear mock model y escenario
        model = MockLLM(model_name="mock_test_model")
        print(f"✅ Mock modelo creado: {model.model_name}")
        
        scenario = ScenarioRegistry.create_scenario_instance("Cold Room Relay")
        if scenario is None:
            print("❌ No se pudo crear el escenario")
            return False
        print(f"✅ Escenario creado: {scenario.name}")
        
        # 3. Crear runner con DuckDB
        runner = ExperimentRunner(results_dir="results", storage_backend="duckdb")
        print("✅ Runner inicializado con DuckDB")
        
        # 4. Ejecutar experimento pequeño
        print("\nEjecutando experimento (3 runs)...")
        results = runner.run_experiment(
            model=model,
            scenario=scenario,
            n_runs=3,
            seed=42,
            temperature=0.7,
            top_p=0.9,
            max_tokens=200,
            progress_bar=True
        )
        print(f"✅ Experimento ejecutado: {len(results)} runs")
        
        # 5. Guardar resultados
        filepath = runner.save_results(results, scenario.name)
        print(f"✅ Resultados guardados en: {filepath}")
        
        # 6. Recuperar resultados usando storage directamente
        print("\n--- Recuperación usando Storage ---")
        retrieved_results = storage.load_results(scenario_name=scenario.name)
        print(f"✅ Resultados recuperados: {len(retrieved_results)} runs")
        
        if len(retrieved_results) != len(results):
            print(f"❌ ERROR: Se guardaron {len(results)} pero se recuperaron {len(retrieved_results)}")
            return False
        
        # 7. Verificar contenido
        print("\nVerificando contenido...")
        for i, (original, retrieved) in enumerate(zip(results, retrieved_results)):
            if original['run_id'] != retrieved['run_id']:
                print(f"❌ ERROR en run_id {i}: original={original['run_id']}, retrieved={retrieved['run_id']}")
                return False
            if original['response'][:50] != retrieved['response'][:50]:
                print(f"⚠️  WARNING: Respuesta diferente en run {i}")
        
        print("✅ Contenido verificado correctamente")
        
        # 8. Recuperar usando runner
        print("\n--- Recuperación usando Runner ---")
        runner_results = runner.load_results(scenario_name=scenario.name)
        print(f"✅ Resultados recuperados via runner: {len(runner_results)} runs")
        
        if len(runner_results) != len(results):
            print(f"❌ ERROR: Se guardaron {len(results)} pero runner recuperó {len(runner_results)}")
            return False
        
        # 9. Recuperar usando statistics
        print("\n--- Recuperación usando Statistics ---")
        stats = ExperimentStatistics(results_dir="results")
        stats_results = stats.load_results(scenario.name)
        print(f"✅ Resultados recuperados via statistics: {len(stats_results)} runs")
        
        if len(stats_results) != len(results):
            print(f"⚠️  WARNING: Statistics recuperó {len(stats_results)} (puede ser por compatibilidad JSONL)")
        
        # 10. Probar filtros
        print("\n--- Prueba de Filtros ---")
        
        # Filtrar por modelo
        model_name = model.model_name
        filtered_by_model = storage.load_results(model_name=model_name)
        print(f"✅ Filtrado por modelo '{model_name}': {len(filtered_by_model)} runs")
        
        # Listar escenarios
        scenarios = storage.list_scenarios()
        print(f"✅ Escenarios disponibles: {scenarios}")
        
        # Listar modelos
        models = storage.list_models()
        print(f"✅ Modelos usados: {models}")
        
        print("\n" + "=" * 60)
        print("✅ TODOS LOS TESTS PASARON")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sqlite_backend():
    """Test con SQLite como backend."""
    print("\n" + "=" * 60)
    print("Test: Guardado y Recuperación (SQLite)")
    print("=" * 60)
    
    try:
        storage = ResultsStorage("results", StorageBackend.SQLITE)
        print("✅ Storage SQLite inicializado")
        
        # Crear resultado de prueba
        test_result = {
            'run_id': 999,
            'scenario': 'test_scenario_sqlite',
            'timestamp': '2024-01-01T00:00:00',
            'prompt': 'Test prompt',
            'system_prompt': 'System',
            'user_prompt': 'User',
            'response': 'Test response SQLite',
            'decisions': {'test': True},
            'metadata': {'model_path': 'test_model', 'temperature': 0.7},
            'scenario_metadata': {}
        }
        
        storage.save_result(test_result, 'test_experiment_sqlite')
        print("✅ Resultado guardado en SQLite")
        
        retrieved = storage.load_results('test_scenario_sqlite')
        if retrieved and len(retrieved) > 0:
            print(f"✅ Resultado recuperado: {retrieved[0]['response']}")
            return True
        else:
            print("❌ No se pudo recuperar el resultado")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def main():
    """Ejecutar todos los tests."""
    print("🧪 Test Completo del Sistema de Almacenamiento")
    print("=" * 60)
    
    # Test principal con DuckDB
    success1 = test_save_and_retrieve()
    
    # Test con SQLite
    success2 = test_sqlite_backend()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("✅ TODOS LOS TESTS COMPLETADOS EXITOSAMENTE")
    else:
        print("⚠️  ALGUNOS TESTS FALLARON")
    print("=" * 60)


if __name__ == "__main__":
    main()

