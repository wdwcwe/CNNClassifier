import copy
import time
import torch
def train_model(model,dataloaders,criterion,optimizer,scheduler,device,num_epoch,is_inception,filename):
    since=time.time()
    best_acc=0   #最好的准确率
    model.to(device)
    val_acc_history=[]
    train_acc_history=[]
    train_losses=[]
    valid_losses=[]
    LRs=[optimizer.param_groups[0]['lr']] #获取当前学习率
    best_model_wts=copy.deepcopy(model.state_dict())#保存最好的模型权重参数
    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch,num_epoch-1))
        print('-'*10)
        #训练和验证
        for phase in ['train','valid']:
            if phase=='train':
                model.train()
            else:
                model.eval()
            running_loss=0.0#该阶段所有批次损失总合
            running_corrects=0#该阶段所有批次预测正确总数
            for inputs ,labels in dataloaders[phase]:
                inputs=inputs.to(device)
                labels=labels.to(device)
                #梯度清0
                optimizer.zero_grad()
                #如果是训练阶段则开启梯度计算,否则不开启节约内存
                with torch.set_grad_enabled(phase=='train'):
                    if is_inception and phase == 'train':  # 针对 Inception 模型的特殊处理。
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs=model(inputs)
                        loss=criterion(outputs,labels)
                    _,preds=torch.max(outputs,1)
                    if phase=='train':
                        loss.backward()
                        optimizer.step()
                running_loss+=loss.item()*inputs.size(0)
                running_corrects+=torch.sum(preds==labels.data)
            epoch_loss=running_loss/len(dataloaders[phase].dataset)
            epoch_acc=running_corrects.double()/len(dataloaders[phase].dataset)
            time_elapsed=time.time()-since
            print("Time elapsed {:.0f}m {:.0f}s".format(time_elapsed//60,time_elapsed%60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase=='valid' and epoch_acc>best_acc:
                best_acc=epoch_acc
                best_model_wts=copy.deepcopy(model.state_dict())
                state={
                    'state_dict':model.state_dict(),
                    'best_acc':best_acc,
                    'optimizer':optimizer.state_dict(),
                }
                torch.save(state,filename)
            if phase=='valid':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
                scheduler.step(epoch_loss)  # 基于损失的变化调整学习率。
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)
        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()
    time_elapsed = time.time() - since
    #打印训练的总时间
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)#通过保存的最佳参数让模型恢复到结果最好的状态
    return  model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs

