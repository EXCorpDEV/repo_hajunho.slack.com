//
// Copyright 2017 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#import "ActionSheetViewController.h"

NSString *const kActionLabelText = @"Actions Verified Here";
NSString *const kActionSheetTitle = @"Action Sheet";
NSString *const kActionSheetSimpleButtonText = @"Simple Button";
NSString *const kActionSheetHideButtonText = @"Hide Button";
NSString *const kActionSheetCancelButtonText = @"Cancel";

@interface ActionSheetViewController ()

@property(strong, nonatomic) IBOutlet UIButton *simpleActionSheetButton;
@property(strong, nonatomic) IBOutlet UIButton *multipleButtonActionSheetButton;
@property(strong, nonatomic) IBOutlet UILabel *actionLabel;

@end

@implementation ActionSheetViewController

- (void)viewDidLoad {
  [super viewDidLoad];

  self.actionLabel.text = kActionLabelText;
  self.simpleActionSheetButton.accessibilityLabel = @"simpleActionSheetButton";
  self.multipleButtonActionSheetButton.accessibilityLabel = @"multipleActionSheetButton";

  [self.simpleActionSheetButton addTarget:self
                                   action:@selector(setupSimpleActionSheet)
                         forControlEvents:UIControlEventTouchUpInside];
  [self.multipleButtonActionSheetButton addTarget:self
                                           action:@selector(setupMultipleButtonActionSheet)
                                 forControlEvents:UIControlEventTouchUpInside];
}

- (void)setupSimpleActionSheet {
  UIAlertController *alertController =
      [UIAlertController alertControllerWithTitle:kActionSheetTitle
                                          message:nil
                                   preferredStyle:UIAlertControllerStyleActionSheet];
  UIAlertAction *cancelAction = [UIAlertAction actionWithTitle:kActionSheetCancelButtonText
                                                         style:UIAlertActionStyleCancel
                                                       handler:^(UIAlertAction *_Nonnull action) {
                                                         self.actionLabel.text = kActionLabelText;
                                                       }];
  UIAlertAction *fooAction = [UIAlertAction actionWithTitle:kActionSheetSimpleButtonText
                                                      style:UIAlertActionStyleDefault
                                                    handler:^(UIAlertAction *_Nonnull action) {
                                                      self.actionLabel.text =
                                                          @"Action Sheet Button Pressed";
                                                    }];
  [alertController addAction:cancelAction];
  [alertController addAction:fooAction];

  CGRect rectInCenterOfMainView =
      CGRectMake(self.view.frame.origin.x, self.view.frame.size.height / 2,
                 self.view.frame.size.width, self.view.frame.size.height);
  alertController.popoverPresentationController.sourceRect = rectInCenterOfMainView;
  alertController.popoverPresentationController.sourceView = self.view;

  [self presentViewController:alertController animated:YES completion:nil];
}

- (void)setupMultipleButtonActionSheet {
  UIAlertController *alertController =
      [UIAlertController alertControllerWithTitle:kActionSheetTitle
                                          message:nil
                                   preferredStyle:UIAlertControllerStyleActionSheet];
  UIAlertAction *cancelAction = [UIAlertAction actionWithTitle:kActionSheetCancelButtonText
                                                         style:UIAlertActionStyleCancel
                                                       handler:^(UIAlertAction *_Nonnull action) {
                                                         self.actionLabel.text = kActionLabelText;
                                                       }];
  UIAlertAction *fooAction = [UIAlertAction actionWithTitle:kActionSheetSimpleButtonText
                                                      style:UIAlertActionStyleDefault
                                                    handler:^(UIAlertAction *_Nonnull action) {
                                                      self.actionLabel.text =
                                                          @"Action Sheet Button Pressed";
                                                    }];
  UIAlertAction *hideAction = [UIAlertAction actionWithTitle:kActionSheetHideButtonText
                                                       style:UIAlertActionStyleDefault
                                                     handler:^(UIAlertAction *_Nonnull action) {
                                                       self.actionLabel.text = @"";
                                                       self.multipleButtonActionSheetButton.hidden =
                                                           YES;
                                                     }];
  [alertController addAction:cancelAction];
  [alertController addAction:fooAction];
  [alertController addAction:hideAction];

  CGRect rectInCenterOfMainView =
      CGRectMake(self.view.frame.origin.x, self.view.frame.size.height / 2,
                 self.view.frame.size.width, self.view.frame.size.height);
  alertController.popoverPresentationController.sourceRect = rectInCenterOfMainView;
  alertController.popoverPresentationController.sourceView = self.view;

  [self presentViewController:alertController animated:YES completion:nil];
}

@end
