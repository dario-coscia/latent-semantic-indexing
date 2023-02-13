python3 benchmark_ir.py --type=svd --dump=svd_250.pkl --embedding_dim=250 
python3 benchmark_ir.py --type=svd --dump=svd_500.pkl --embedding_dim=500
python3 benchmark_ir.py --type=svd --dump=svd_750.pkl --embedding_dim=750 
python3 benchmark_ir.py --type=svd --dump=svd_1000.pkl --embedding_dim=1000

python3 benchmark_ir.py --type=kpca --dump=kpca_rbf_250.pkl --embedding_dim=250 --kernel=rbf
python3 benchmark_ir.py --type=kpca --dump=kpca_rbf_500.pkl --embedding_dim=500 --kernel=rbf
python3 benchmark_ir.py --type=kpca --dump=kpca_rbf_750.pkl --embedding_dim=750 --kernel=rbf
python3 benchmark_ir.py --type=kpca --dump=kpca_rbf_1000.pkl --embedding_dim=1000 --kernel=rbf


python3 benchmark_ir.py --type=kpca --dump=kpca_poly_250.pkl --embedding_dim=250 --kernel=poly  --gamma=2
python3 benchmark_ir.py --type=kpca --dump=kpca_poly_500.pkl --embedding_dim=500 --kernel=poly  --gamma=2
python3 benchmark_ir.py --type=kpca --dump=kpca_poly_750.pkl --embedding_dim=750 --kernel=poly --gamma=2
python3 benchmark_ir.py --type=kpca --dump=kpca_poly_1000.pkl --embedding_dim=1000 --kernel=poly  --gamma=2

python3 benchmark_ir.py --type=kpca --dump=kpca_tanh_250.pkl --embedding_dim=250 --kernel=tanh 
python3 benchmark_ir.py --type=kpca --dump=kpca_tanh_500.pkl --embedding_dim=500 --kernel=tanh 
python3 benchmark_ir.py --type=kpca --dump=kpca_tanh_750.pkl --embedding_dim=750 --kernel=tanh 
python3 benchmark_ir.py --type=kpca --dump=kpca_tanh_1000.pkl --embedding_dim=1000 --kernel=tanh 

python3 benchmark_ir.py --type=ae --dump=ae_250.pkl --embedding_dim=250 
python3 benchmark_ir.py --type=ae --dump=ae_500.pkl --embedding_dim=500
python3 benchmark_ir.py --type=ae --dump=ae_750.pkl --embedding_dim=750 
python3 benchmark_ir.py --type=ae --dump=ae_1000.pkl --embedding_dim=1000