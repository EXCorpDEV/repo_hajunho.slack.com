//
//  ratioNtype.swift
//  eograph
//
//  Created by Junho HA on 02/22/2022.
//  Copyright © 2022 eoflow. All rights reserved.
//

import UIKit

open class ratioNtype {
    public var ratio : CGFloat = 0.0
    public var type : graphType = .BAR
    //    var datas : hjh
    public init(ratio: CGFloat, type: graphType) {
        self.ratio = ratio
        self.type = type
    }
}
