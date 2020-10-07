//
// Copyright 2018 Google Inc.
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

#import "VisibilityTestViewController.h"

@implementation VisibilityTestViewController

- (void)viewDidLoad {
  [super viewDidLoad];

  CGFloat halfPixelInPoint = 1.0f / (2.0f * [UIScreen mainScreen].scale);

  [self addUnalignedViewWithAccessibilityID:@"unalignedPixel1"
                                      frame:CGRectMake(50 + halfPixelInPoint, 120, 1, 1)];
  [self addUnalignedViewWithAccessibilityID:@"unalignedPixel2"
                                      frame:CGRectMake(50, 121 + halfPixelInPoint, 1, 1)];
  [self addUnalignedViewWithAccessibilityID:@"unalignedPixel3"
                                      frame:CGRectMake(52 + halfPixelInPoint,
                                                       121 + halfPixelInPoint, 1, 1)];
  [self addUnalignedViewWithAccessibilityID:@"unalignedPixel4"
                                      frame:CGRectMake(52, 120, 1 + halfPixelInPoint,
                                                       1 + halfPixelInPoint)];
  [self addUnalignedViewWithAccessibilityID:@"unalignedPixelWithOnePixelSize"
                                      frame:CGRectMake(57 + halfPixelInPoint, 120,
                                                       2 * halfPixelInPoint, 2 * halfPixelInPoint)];
  [self addUnalignedViewWithAccessibilityID:@"unalignedPixelWithHalfPixelSize"
                                      frame:CGRectMake(55, 120 + halfPixelInPoint, halfPixelInPoint,
                                                       halfPixelInPoint)];
  [self addUnalignedViewWithAccessibilityID:@"unalignedPixelWithFractionPixelSize"
                                      frame:CGRectMake(70, 90, 5, halfPixelInPoint * 1.2f)];

  UITapGestureRecognizer *tap =
      [[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(hideButton)];
  [self.button addGestureRecognizer:tap];
  self.button.accessibilityIdentifier = @"VisibilityButton";

  self.translucentLabel.accessibilityIdentifier = @"translucentLabel";
  self.translucentOverlappingView.accessibilityIdentifier = @"translucentOverlappingView";
  self.leftView.accessibilityIdentifier = @"AView1";
  self.centerView.accessibilityIdentifier = @"AView2";
  self.rightView.accessibilityIdentifier = @"AView3";
  self.bottomScrollView.accessibilityIdentifier = @"bottomScrollView";
  self.coverScrollView.accessibilityIdentifier = @"coverScrollView";

  [self.imageView setBackgroundColor:[UIColor greenColor]];

  UIVisualEffect *blurEffect = [UIBlurEffect effectWithStyle:UIBlurEffectStyleDark];
  UIVisualEffectView *visualEffectView = [[UIVisualEffectView alloc] initWithEffect:blurEffect];
  visualEffectView.frame = self.imageView.bounds;
  visualEffectView.accessibilityIdentifier = @"visualEffectsImageView";
  [self.imageView addSubview:visualEffectView];
  // Check visibility of orangeView to invoke thorough visibility checker.
  self.orangeView.accessibilityIdentifier = @"orangeView";
  // Purple view is drawn on top of orange view, and purple view is transformed so that the quick
  // visibility checker will fallback to use thorough visibility checker for orangeView.
  self.purpleView.transform = CGAffineTransformMakeRotation(M_PI_4);
}

- (void)viewDidLayoutSubviews {
  [super viewDidLayoutSubviews];

  self.bottomScrollView.contentSize = CGSizeMake(1000, 1000);
  self.coverScrollView.contentSize = CGSizeMake(1000, 1000);

  self.bottomScrollView.contentOffset = CGPointMake(100, 100);
  self.coverScrollView.contentOffset = CGPointMake(100, 100);
}

#pragma mark - Private

- (void)addUnalignedViewWithAccessibilityID:(NSString *)accID frame:(CGRect)frame {
  UIView *view = [[UIView alloc] initWithFrame:frame];
  view.backgroundColor = [UIColor blackColor];
  view.accessibilityIdentifier = accID;
  [self.view addSubview:view];
}

- (IBAction)toggleActivationPoint:(UISwitch *)sender {
  self.activationPointCover.hidden = sender.isOn;
}

- (void)hideButton {
  // Obscure the button with another view.
  UIView *view = [[UIView alloc] initWithFrame:self.button.frame];
  [view setBackgroundColor:[UIColor grayColor]];
  [self.button.superview addSubview:view];
  [self.button.superview setNeedsDisplay];
}

@end
