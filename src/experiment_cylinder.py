"""
Main experiment script for flow past a cylinder
"""
import sys
import os
import torch
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.neural_networks import create_model
from src.solvers.standard_pinn import CylinderPINN
from src.solvers.curriculum_pinn import CurriculumCylinderPINN
from src.numerics.fdm_cylinder import CylinderFDM
from src.visualization.cylinder_plots import create_all_cylinder_plots
from src.configs.cylinder_config import get_config


def get_fdm_cache_path(config, cache_dir='cache/fdm'):
    """Generate unique cache filepath for FDM solution"""
    Re = config['Re_target']
    nx = config['fdm_nx']
    ny = config['fdm_ny']
    dt = config['fdm_dt']
    max_iter = config['fdm_max_iter']
    
    filename = f"fdm_Re{Re}_grid{nx}x{ny}_dt{dt}_iter{max_iter}.pkl"
    return os.path.join(cache_dir, filename)


def run_cylinder_experiment(config_name='default', output_dir='outputs/cylinder',
                           use_data_loss=False, use_fdm_cache=True, force_recompute_fdm=False):
    """
    Run complete cylinder flow experiment
    
    Args:
        config_name: Configuration name
        output_dir: Output directory for results
        use_data_loss: Whether to use data loss
        use_fdm_cache: Whether to use cached FDM solution
        force_recompute_fdm: Force recompute FDM even if cache exists
    """
    
    config = get_config(config_name)
    
    # Override data loss setting if specified
    if use_data_loss:
        config['use_data_loss'] = True
        config.setdefault('N_data', 1000)
        config.setdefault('lambda_data', 10.0)
        print(f"\n⚠ Data loss ENABLED with {config['N_data']} sampling points")
    
    print("\n" + "="*80)
    print(f"FLOW PAST CYLINDER EXPERIMENT: {config_name}")
    print("="*80)
    print("\nConfiguration:")
    config_to_print = {k: v for k, v in config.items() if k != 'fdm_solver'}
    print(json.dumps(config_to_print, indent=2))
    print("="*80 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # ==================== STEP 1: FDM Ground Truth ====================
    print("="*80)
    print("STEP 1: Generating Ground Truth with FDM")
    print("="*80)
    
    fdm_cache_path = get_fdm_cache_path(config)
    
    if use_fdm_cache and not force_recompute_fdm and os.path.exists(fdm_cache_path):
        print(f"\n📦 Found cached FDM solution: {fdm_cache_path}")
        print("="*80)
        fdm_solver = CylinderFDM.load_solution(fdm_cache_path, config)
        print("="*80)
    else:
        if force_recompute_fdm:
            print("\n⚠ Force recomputing FDM solution (ignoring cache)")
        else:
            print("\n💻 No cached solution found, computing FDM solution...")
        
        print("="*80)
        fdm_solver = CylinderFDM(
            Re=config['Re_target'],
            U_inf=config['U_inf'],
            cylinder_center=config['cylinder_center'],
            cylinder_radius=config['cylinder_radius'],
            nx=config['fdm_nx'],
            ny=config['fdm_ny'],
            x_domain=config['x_domain'],
            y_domain=config['y_domain'],
            dt=config['fdm_dt'],
            max_iter=config['fdm_max_iter'],
            tol=config['fdm_tolerance'],
            config=config
        )
        fdm_solver.solve()
        
        # Save to cache
        if use_fdm_cache:
            print(f"\n💾 Saving FDM solution to cache...")
            fdm_solver.save_solution(fdm_cache_path)
        
        print("="*80)
    
    # ==================== STEP 2: Standard PINN ====================
    print("\n" + "="*80)
    print("STEP 2: Training Standard PINN")
    print("="*80)
    
    if config['model_type'] == 'mlp':
        model_std = create_model(model_type='mlp', layers=config['layers'], 
                                activation=config['activation'])
    elif config['model_type'] == 'resnet':
        model_std = create_model(model_type='resnet', 
                                input_dim=config['input_dim'],
                                hidden_dim=config['hidden_dim'],
                                output_dim=config['output_dim'],
                                num_blocks=config.get('num_blocks', 4),
                                activation=config['activation'])
    
    standard_config = {
        'Re': config['Re_target'],
        'U_inf': config['U_inf'],
        'cylinder_center': config['cylinder_center'],
        'cylinder_radius': config['cylinder_radius'],
        'x_domain': config['x_domain'],
        'y_domain': config['y_domain'],
        'boundary_conditions': config['boundary_conditions'],
        'initial_conditions': config['initial_conditions'],
        'N_bc_inlet': config['N_bc_inlet'],
        'N_bc_outlet': config['N_bc_outlet'],
        'N_bc_wall': config['N_bc_wall'],
        'N_bc_cylinder': config['N_bc_cylinder'],
        'N_pde': config['N_pde'],
        'lambda_bc_inlet': config['lambda_bc_inlet'],
        'lambda_bc_outlet': config['lambda_bc_outlet'],
        'lambda_bc_wall': config['lambda_bc_wall'],
        'lambda_bc_cylinder': config['lambda_bc_cylinder'],
        'lambda_pde': config['lambda_pde'],
        'lambda_cont': config['lambda_cont'],
        'lambda_data': config['lambda_data'],
        'epochs': config['pinn_epochs'],
        'learning_rate': config['pinn_learning_rate'],
        'optimizer': config.get('pinn_optimizer', 'adam'),
        'use_scheduler': config.get('pinn_use_scheduler', True),
        'lr_decay_steps': config.get('pinn_lr_decay_steps', 5000),
        'lr_decay_rate': config.get('pinn_lr_decay_rate', 0.95),
        'use_data_loss': config['use_data_loss'],
        'N_data': config.get('N_data', 1000),
        'fdm_solver': fdm_solver if config['use_data_loss'] else None,
    }
    
    pinn_standard = CylinderPINN(model_std, standard_config, device=device)
    pinn_standard.train(verbose=True, save_interval=1000)
    
    # ==================== STEP 3: Curriculum PINN ====================
    print("\n" + "="*80)
    print("STEP 3: Training Curriculum PINN")
    print("="*80)
    
    if config['model_type'] == 'mlp':
        model_curr = create_model(model_type='mlp', layers=config['layers'], 
                                 activation=config['activation'])
    elif config['model_type'] == 'resnet':
        model_curr = create_model(model_type='resnet', 
                                 input_dim=config['input_dim'],
                                 hidden_dim=config['hidden_dim'],
                                 output_dim=config['output_dim'],
                                 num_blocks=config.get('num_blocks', 4),
                                 activation=config['activation'])
    
    curriculum_config = {
        **standard_config,
        'curriculum_stages': config['curriculum_stages'],
        'learning_rate': config['curriculum_learning_rate'],
    }
    
    pinn_curriculum = CurriculumCylinderPINN(model_curr, curriculum_config, device=device)
    pinn_curriculum.train(verbose=True, save_interval=500)
    
    # ==================== STEP 4: Visualization ====================
    print("\n" + "="*80)
    print("STEP 4: Generating Visualizations")
    print("="*80)
    
    create_all_cylinder_plots(fdm_solver, pinn_standard, pinn_curriculum,
                             config, output_dir=output_dir)
    
    # ==================== STEP 5: Save Results ====================
    os.makedirs(output_dir, exist_ok=True)
    
    pinn_standard.save_model(os.path.join(output_dir, 'standard_pinn.pt'))
    pinn_curriculum.save_model(os.path.join(output_dir, 'curriculum_pinn.pt'))
    
    config_to_save = {k: v for k, v in config.items() if k != 'fdm_solver'}
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config_to_save, f, indent=2)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    if config['use_data_loss']:
        print(f"✓ Data loss was ENABLED with {config.get('N_data', 1000)} sampling points")
    if use_fdm_cache:
        print(f"✓ FDM cache: {fdm_cache_path}")
    print("="*80 + "\n")
    
    return fdm_solver, pinn_standard, pinn_curriculum


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run cylinder flow experiment')
    parser.add_argument('--config', type=str, default='low_re_light',
                       help='Configuration (default, low_re_light, low_re, high_re, etc.)')
    parser.add_argument('--output', type=str, default='outputs/cylinder',
                       help='Output directory')
    parser.add_argument('--use-data-loss', action='store_true',
                       help='Enable data loss during training')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable FDM caching (always recompute)')
    parser.add_argument('--force-fdm', action='store_true',
                       help='Force recompute FDM even if cache exists')
    
    args = parser.parse_args()
    
    run_cylinder_experiment(
        config_name=args.config,
        output_dir=args.output,
        use_data_loss=args.use_data_loss,
        use_fdm_cache=not args.no_cache,
        force_recompute_fdm=args.force_fdm
    )