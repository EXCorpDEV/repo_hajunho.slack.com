//
//  jhscene_p.swift
//  bridge8
//
//  Created by Junho HA on 2022. 2. 22.
//  Copyright © 2022년 eoflow. All rights reserved.
//
// CAUTION! All the primitive type of jhChart is x100 percentage like a 0~10000 not 0~100%
//

//Device  Native Resolution(Pixels)  UIKit Size (Points)    Native Scale factor    UIKit Scale factor
//iphone XS Max     1125 x 2436      414 x 896
//iphone XS         1242 x 2688      375 x 812
//iPhone XR          828 x 1792      414 x 896
//iPhone X          1125 x 2436      375 x 812                    3.0      3.0
//iPhone 8 Plus     1080 x 1920      414 x 736                    2.608    3.0
//iPhone 8           750 x 1334      375 x 667                    2.0      2.0
//iPhone 7 Plus     1080 x 1920      414 x 736                    2.608    3.0
//iPhone 6s Plus    1080 x 1920      375 x 667                    2.608    3.0
//iPhone 6 Plus     1080 x 1920      375 x 667                    2.608    3.0
//iPhone 7           750 x 1334      375 x 667                    2.0      2.0
//iPhone 6s          750 x 1334      375 x 667                    2.0      2.0
//iPhone 6           750 x 1334      375 x 667                    2.0      2.0
//iPhone SE          640 x 1136      320 x 568                    2.0      2.0
//iPad Pro 12.9-inch
//(2nd generation)   2048 x 2732    1024 x 1366                   2.0     2.0
//iPad Pro 10.5-inch 2224 x 1668    1112 x 834                    2.0     2.0
//iPad Pro (12.9)    2048 x 2732    1024 x 1366                   2.0     2.0
//iPad Pro (9.7-inch)1536 x 2048     768 x 1024                   2.0     2.0
//iPad Air 2         1536 x 2048     768 x 1024                   2.0     2.0
//iPad Mini 4        1536 x 2048     768 x 1024                   2.0     2.0

import UIKit

protocol jhPanel_p {
    var jhEnforcingMode : Bool { get set }
    var jhPanelID : Int { get set }
//    var jhSceneFrameWidth : CGFloat { get set }
//    var jhSceneFrameHeight : CGFloat { get set }
    mutating func jhReSize(size : CGSize)
//    func getData() -> Array<CGFloat>
    func drawDatas()
    func jhRedraw()
    func drawAxes()
}

protocol AnyEquatable {
    func equals(rhs: AnyEquatable) -> Bool
    func canEqualReverseDispatch(lhs: AnyEquatable) -> Bool
}

/*final*/ func ==(lhs: AnyEquatable, rhs: AnyEquatable) -> Bool {
    return lhs.equals(rhs: rhs) // Fix the type of the LHS using dynamic dispatch.
}
/*final*/ func !=(lhs: AnyEquatable, rhs: AnyEquatable) -> Bool {
    return !lhs.equals(rhs: rhs) // Fix the type of the LHS using dynamic dispatch.
}
