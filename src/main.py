import torch
from src.data.cremad_dataset import load_cremad
from src.models.emotion_classifier import EmotionClassifier
from src.utils.training_utils import train_epoch, eval_model

def main(config):
    print(config)
    torch.manual_seed(config['random_seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data
    train_dataset, dev_dataset, test_dataset = load_cremad(config)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

    # Initialize model
    model = EmotionClassifier(config).to(device)

    # Define optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['lr_decay_step'], config['lr_decay_ratio'])

    # Training loop
    best_dev_f1 = 0.0
    for epoch in range(config['epochs']):
        train_epoch(config, epoch, model, device, train_dataloader, optimizer, scheduler)
        loss, f1 = eval_model(config, model, device, dev_dataloader)
        print(f"Epoch: {epoch}, Dev Loss: {loss:.4f}, Dev F1: {f1:.4f}")
        if f1 > best_dev_f1:
                best_dev_f1 = f1
                best_state = best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
        
        # Best state
        model.load_state_dict(best_state)
        print('Best model loaded at epoch {} with dev F1: {}'.format(best_epoch, best_dev_f1))
        print("Start testing on test dataset ...")
        _, f1 = eval(args, model, device, test_dataloader, test=True)
        if not os.path.exists(args.ckpt_path):
            os.mkdir(args.ckpt_path)

        model_name = 'best_model_of_dataset_{}_' \
                        'optimizer_{}_' \
                        'epoch_{}_f1_{}.pth'.format(args.dataset,
                                                    args.optimizer,
                                                    epoch, f1)

        saved_dict = {'saved_epoch': epoch,
                        'fusion': args.fusion_method,
                        'f1': f1,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()}

        save_dir = os.path.join(args.ckpt_path, model_name)

        torch.save(saved_dict, save_dir)
        print('The best model has been saved at {}.'.format(save_dir))

                

    else:
        loaded_dict = torch.load(args.ckpt_path)
        fusion = loaded_dict['fusion']
        state_dict = loaded_dict['model']

        assert fusion == args.fusion_method, 'inconsistency between fusion method of loaded model and args !'

        model.load_state_dict(state_dict)
        print('Trained model loaded! Testing ...')

        loss, f1 = eval(args, model, device, test_dataloader, test=True)
        


if __name__ == "__main__":
    main()