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

#import "MainViewController.h"

#import "AccessibilityViewController.h"
#import "ActionSheetViewController.h"
#import "ActivityIndicatorViewController.h"
#import "AlertViewController.h"
#import "AnimationViewController.h"
#import "BasicViewController.h"
#import "CollectionViewController.h"
#import "GestureViewController.h"
#import "ImageViewController.h"
#import "LayoutViewController.h"
#import "MultiFingerSwipeGestureRecognizerViewController.h"
#import "NetworkTestViewController.h"
#import "PickerViewController.h"
#import "PresentedViewController.h"
#import "RotatedViewsViewController.h"
#import "ScrollViewController.h"
#import "SimpleTapViewController.h"
#import "SliderViewController.h"
#import "SystemAlertsViewController.h"
#import "TableViewController.h"
#import "TypingViewController.h"
#import "VisibilityTestViewController.h"
#import "WKWebViewController.h"
#import "ZoomingScrollViewController.h"

static NSString *gTableViewIdentifier = @"TableViewIdentifier";

@interface MainViewController () <UITableViewDataSource, UITableViewDelegate, UIActionSheetDelegate>
@end

@implementation MainViewController {
  NSDictionary *_nameToControllerMap;
}

- (id)initWithNibName:(NSString *)nibNameOrNil bundle:(NSBundle *)nibBundleOrNil {
  self = [super initWithNibName:nibNameOrNil bundle:nibBundleOrNil];
  if (self) {
    self.title = @"EarlGrey TestApp";
    // TODO: Clean this up so we have text to selector mapping instead of text to class
    // and text to [NSNull null] in some cases.
    _nameToControllerMap = @{
      @"Accessibility Views" : [AccessibilityViewController class],
      @"Action Sheets" : [ActionSheetViewController class],
      @"Activity Indicator Views" : [ActivityIndicatorViewController class],
      @"Alert Views" : [AlertViewController class],
      @"Animations" : [AnimationViewController class],
      @"Basic Views" : [BasicViewController class],
      @"Collection Views" : [CollectionViewController class],
      @"Gesture Tests" : [GestureViewController class],
      @"Layout Tests" : [LayoutViewController class],
      @"Pinch Tests" : [ImageViewController class],
      @"Network Test" : [NetworkTestViewController class],
      @"Picker Views" : [PickerViewController class],
      @"Presented Views" : [PresentedViewController class],
      @"Rotated Views" : [RotatedViewsViewController class],
      @"Scroll Views" : [ScrollViewController class],
      @"Simple Tap View" : [SimpleTapViewController class],
      @"Slider Views" : [SliderViewController class],
      @"System Alerts" : [SystemAlertsViewController class],
      @"Table Views" : [TableViewController class],
      @"Typing Views" : [TypingViewController class],
      @"Visibility Tests" : [VisibilityTestViewController class],
      @"Multi finger swipe gestures" : [MultiFingerSwipeGestureRecognizerViewController class],
      @"Zooming Scroll View" : [ZoomingScrollViewController class],
      @"WKWebView" : [WKWebViewController class],
    };
  }
  return self;
}

- (void)viewDidLoad {
  [super viewDidLoad];

  self.tableview.delegate = self;
  self.tableview.dataSource = self;

  // Making the nav bar not translucent so it won't cover UI elements.
  [self.navigationController.navigationBar setTranslucent:NO];
}

// If we find that the orientation of the device / simulator is not
// UIDeviceOrientationPortrait, then for testing purposes, we rotate
// it to UIDeviceOrientationPortrait. However, the simulator itself
// tries to correct the orientation since we support all orientations
// in our test app. This removes the automated orientation correction.
- (BOOL)shouldAutorotate {
  return NO;
}

#pragma mark - UITableViewDataSource

- (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)section {
  NSAssert(section == 0, @"We have more than one section?");
  return (NSInteger)[_nameToControllerMap count];
}

- (UITableViewCell *)tableView:(UITableView *)tableView
         cellForRowAtIndexPath:(NSIndexPath *)indexPath {
  UITableViewCell *cell = [tableView dequeueReusableCellWithIdentifier:gTableViewIdentifier];

  if (cell == nil) {
    cell = [[UITableViewCell alloc] initWithStyle:UITableViewCellStyleDefault
                                  reuseIdentifier:gTableViewIdentifier];
  }

  cell.selectionStyle = UITableViewCellSelectionStyleNone;
  NSArray *allKeys = [_nameToControllerMap.allKeys
      sortedArrayUsingSelector:@selector(localizedCaseInsensitiveCompare:)];
  NSString *key = [allKeys objectAtIndex:(NSUInteger)indexPath.row];
  cell.textLabel.text = key;
  return cell;
}

#pragma mark - UITableViewDelegate

- (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath {
  NSArray *allKeys = [_nameToControllerMap.allKeys
      sortedArrayUsingSelector:@selector(localizedCaseInsensitiveCompare:)];
  NSString *key = [allKeys objectAtIndex:(NSUInteger)indexPath.row];
  Class viewController = _nameToControllerMap[key];
  UIViewController *vc =
      [[viewController alloc] initWithNibName:NSStringFromClass(viewController) bundle:nil];

  if ([key isEqualToString:@"Presented Views"]) {
    [self.navigationController presentViewController:vc animated:NO completion:nil];
  } else {
    [UIView beginAnimations:nil context:NULL];
    [UIView setAnimationCurve:UIViewAnimationCurveEaseInOut];
    [UIView setAnimationDuration:0.2];
    [self.navigationController pushViewController:vc animated:NO];
    [UIView setAnimationTransition:UIViewAnimationTransitionFlipFromRight
                           forView:self.navigationController.view
                             cache:NO];
    [UIView commitAnimations];
  }
}

@end
